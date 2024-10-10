# Khmer-RNN: Khmer to English Neural Machine Translation Example

This project implements a Sequence-to-Sequence (Seq2Seq) model with attention for translating Khmer sentences to English. The implementation is based on the PyTorch Seq2Seq Translation Tutorial with modifications for the Khmer-English language pair. From Pytorch tutorial, the encoder is quite efficient for French because the words is somehow written in English letter. 

## Project Structure

- `train.py`: Script for training the model
- `inspect_model.py`: Script to inspect the trained model's weights and parameters
- `inference.py`: Script for running inference on the trained model
- `tok155129.model`: SentencePiece tokenizer model
- `khm-eng/`: Directory containing the dataset

## Model Architecture

The model consists of an Encoder-Decoder architecture with attention:

- Encoder: GRU (Gated Recurrent Unit)
- Decoder: GRU with Bahdanau Attention
- Embedding: Separate embedding layers for input and output languages

## Training

To train the model, run: python train.py

The script loads the SentencePiece tokenizer, prepares the dataset, and trains the model. The trained model is saved as `translation_model.pth`.

## Model Inspection

To inspect the trained model, run: python inspect_model.py

## Inference

This script allows you to input Khmer sentences and get their English translations. It also includes functions for batch translation and BLEU score calculation.

Key components of the inference process: python: khmer-rnn/inference.py

# Dataset

The dataset is located in the `khm-eng/` directory. It contains parallel Khmer-English sentences used for training and evaluation.

# Dependencies
pip install -r requirements.txt

## Future Improvements

- Implement beam search for better translation quality
- Experiment with different model architectures (e.g., Transformer)
- Expand the dataset for improved performance
- Implement more comprehensive evaluation metrics

## License

[MIT License](LICENSE)

## Acknowledgements

This project is based on the PyTorch Seq2Seq Translation Tutorial and adapted for Khmer-English translation.

The original source: https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html