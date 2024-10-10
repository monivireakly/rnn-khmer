import torch
import sentencepiece as spm

sp = spm.SentencePieceProcessor()
sp.load("tok155129.model")

# Load the saved model
class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sp.encode_as_pieces(sentence):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1
            
checkpoint = torch.load('translation_model.pth', map_location=torch.device('cpu'))

# Print the keys in the checkpoint
print("Keys in the checkpoint:")
for key in checkpoint.keys():
    print(f"- {key}")

# Print information about the encoder and decoder state dictionaries
print("\nEncoder state dict:")
for key, value in checkpoint['encoder_state_dict'].items():
    print(f"- {key}: {value.shape}")

print("\nDecoder state dict:")
for key, value in checkpoint['decoder_state_dict'].items():
    print(f"- {key}: {value.shape}")

# Print other relevant information
print(f"\nHidden size: {checkpoint['hidden_size']}")
print(f"Input language n_words: {checkpoint['input_lang'].n_words}")
print(f"Output language n_words: {checkpoint['output_lang'].n_words}")
