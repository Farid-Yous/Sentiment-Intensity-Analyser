# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 01:05:21 2024

@author: farid
"""
import re
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import pickle
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"   

def preprocess_data(text):
    # Lowercase the text
    text = text.lower()
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    return text

max_text_length = 200

# Load the pretrained model
model = load_model('Sentiment.keras')

# Load the previously saved tokenizer
with open('tokenizer.pkl', 'rb') as file:
    tokenizer = pickle.load(file)

# Summarize model structure
model.summary()

# Define the sample text for prediction
sample_texts = [""]

# Preprocess the text
sample_texts_preprocessed = [preprocess_data(text) for text in sample_texts]

# Convert text to sequences
sample_seqs = tokenizer.texts_to_sequences(sample_texts_preprocessed)

# Pad the sequences to ensure uniform length
sample_padded = pad_sequences(sample_seqs, maxlen=max_text_length, padding='post')

# Make predictions
predictions = model.predict(sample_padded)

# Output the sentiment scores
for i, prediction in enumerate(predictions):
    print(f"Text: {sample_texts[i]}")
    print(f"Sentiment score: {prediction[0]}")

