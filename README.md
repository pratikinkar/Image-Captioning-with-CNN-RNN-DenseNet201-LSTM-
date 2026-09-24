# Image Captioning with CNN-RNN (DenseNet201 + LSTM)

A hybrid deep learning pipeline that generates natural-language captions for images,
combining a pretrained CNN for visual feature extraction with an LSTM-based
sequence model for language generation.

## Overview

- **Vision side:** DenseNet201 (pretrained on ImageNet) extracts a 1920-dimensional
  feature vector per image via transfer learning.
- **Language side:** Captions are cleaned, tokenized, and represented with a Keras
  `Embedding` layer; an LSTM predicts the next word in the sequence.
- **Architecture:** image features and word embeddings are merged and decoded
  word-by-word, in the style of the "Show and Tell" image captioning approach.
- **Training:** a custom Keras `Sequence` data generator streams batches of
  (image feature, partial caption) → (next word) pairs for memory-efficient
  training, with early stopping, learning-rate reduction, and checkpointing.

## Dataset

[Flickr8k](https://www.kaggle.com/datasets/adityajn105/flickr8k) — 8,000 images,
each with 5 human-written captions.

To run this notebook:
1. Download the dataset from Kaggle (link above) and unzip it.
2. Place it so the folder structure looks like:
   ```
   data/flickr8k/
     captions.txt
     Images/
       *.jpg
   ```
3. Update `BASE_DIR` in the config cell near the top of the notebook if your
   path differs.

## Tech Stack

Python · TensorFlow / Keras · DenseNet201 · LSTM · NumPy · Pandas · Matplotlib

## Workflow

1. Load and clean captions (lowercase, strip punctuation, add start/end tokens)
2. Tokenize captions and build vocabulary
3. Extract image features with pretrained DenseNet201
4. Train the CNN-RNN captioning model with a custom data generator
5. Evaluate via training/validation loss curves
6. Generate captions for unseen test images via greedy decoding

## Results

The notebook plots training/validation loss curves and displays sample
generated captions alongside their corresponding test images for qualitative
evaluation.

## Future Improvements

- Attention-based decoding (Show, Attend and Tell)
- Quantitative evaluation with BLEU / METEOR scores
- Training on larger datasets (Flickr30k, MS COCO)

## Author

**Pratik Inkar**
[LinkedIn](https://www.linkedin.com/in/pratikinkar/) · [GitHub](https://github.com/pratikinkar)
