import torch
import torch.nn as nn
import torch.nn.functional as F
import sentencepiece as spm
from tqdm import tqdm
import numpy as np

# Constants
SOS_token = 0
EOS_token = 1
MAX_LENGTH = 20
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load SentencePiece model
sp = spm.SentencePieceProcessor()
sp.load("tok155129.model")

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_p=0.1):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, input):
        embedded = self.dropout(self.embedding(input))
        output, hidden = self.gru(embedded)
        return output, hidden

class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super(BahdanauAttention, self).__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size)
        self.Ua = nn.Linear(hidden_size, hidden_size)
        self.Va = nn.Linear(hidden_size, 1)

    def forward(self, query, keys):
        scores = self.Va(torch.tanh(self.Wa(query) + self.Ua(keys)))
        scores = scores.squeeze(2).unsqueeze(1)
        weights = F.softmax(scores, dim=-1)
        context = torch.bmm(weights, keys)
        return context, weights

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1):
        super(AttnDecoderRNN, self).__init__()
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.attention = BahdanauAttention(hidden_size)
        self.gru = nn.GRU(2 * hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None):
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=device).fill_(SOS_token)
        decoder_hidden = encoder_hidden
        decoder_outputs = []
        attentions = []

        for i in range(MAX_LENGTH):
            decoder_output, decoder_hidden, attn_weights = self.forward_step(
                decoder_input, decoder_hidden, encoder_outputs
            )
            decoder_outputs.append(decoder_output)
            attentions.append(attn_weights)

            _, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze(-1).detach()

            if decoder_input.item() == EOS_token:
                break

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        attentions = torch.cat(attentions, dim=1)

        return decoder_outputs, decoder_hidden, attentions

    def forward_step(self, input, hidden, encoder_outputs):
        embedded = self.dropout(self.embedding(input))
        query = hidden.permute(1, 0, 2)
        context, attn_weights = self.attention(query, encoder_outputs)
        input_gru = torch.cat((embedded, context), dim=2)
        output, hidden = self.gru(input_gru, hidden)
        output = self.out(output)
        return output, hidden, attn_weights

def normalize_string(s):
    return ' '.join(sp.encode_as_pieces(s))

def indexes_from_sentence(lang, sentence):
    return [lang.word2index.get(word, 0) for word in sp.encode_as_pieces(sentence)] + [EOS_token]

def tensor_from_sentence(lang, sentence):
    indexes = indexes_from_sentence(lang, sentence)
    return torch.tensor(indexes, dtype=torch.long, device=device).unsqueeze(0)

def translate(encoder, decoder, sentence, input_lang, output_lang, max_length=MAX_LENGTH):
    encoder.eval()
    decoder.eval()

    with torch.no_grad():
        input_tensor = tensor_from_sentence(input_lang, sentence)
        encoder_outputs, encoder_hidden = encoder(input_tensor)

        decoder_outputs, _, attentions = decoder(encoder_outputs, encoder_hidden)

        _, topi = decoder_outputs.topk(1)
        decoded_words = []

        for i in range(max_length):
            token = topi[0, i].item()
            if token == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[token])

    return decoded_words, attentions

# Load the saved model
checkpoint = torch.load('translation_model.pth', map_location=torch.device('cpu'))

input_lang = checkpoint['input_lang']
output_lang = checkpoint['output_lang']
hidden_size = checkpoint['hidden_size']

encoder = EncoderRNN(input_lang.n_words, hidden_size).to(device)
decoder = AttnDecoderRNN(hidden_size, output_lang.n_words).to(device)

encoder.load_state_dict(checkpoint['encoder_state_dict'])
decoder.load_state_dict(checkpoint['decoder_state_dict'])

encoder.eval()
decoder.eval()

# Function to translate a sentence
def translate_sentence(sentence):
    normalized_sentence = normalize_string(sentence)
    output_words, attentions = translate(encoder, decoder, normalized_sentence, input_lang, output_lang)
    
    translated = ' '.join(output_words[:-1])
    # Remove the leading space and replace remaining ▁ with spaces
    translated = translated.strip().replace(' ▁', ' ').replace('▁', '')
    return translated

# Function to calculate BLEU score (simplified version)
def calculate_bleu(reference, candidate):
    return sum(1 for w in candidate if w in reference) / len(candidate) if len(candidate) > 0 else 0

# Function to translate and evaluate all sentences in the dataset
def evaluate_translations(input_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    sources, targets = [], []
    bleu_scores = []
    lengths = []

    for line in tqdm(lines, desc="Translating"):
        parts = line.strip().split('\t')
        if len(parts) == 2:
            source, target = parts
            sources.append(source)
            targets.append(target)
            
            translation = translate_sentence(source)
            bleu = calculate_bleu(target.split(), translation.split())
            bleu_scores.append(bleu)
            lengths.append(len(source.split()))

    return sources, targets, bleu_scores, lengths

# Example usage
while True:
    input_sentence = input("Enter a sentence in Khmer (or 'q' to quit): ")
    if input_sentence.lower() == 'q':
        break
    translated_sentence = translate_sentence(input_sentence)
    print(f"Translation: {translated_sentence}")
    