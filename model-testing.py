# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 01:05:21 2024

@author: farid
"""
import nltk
import re
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from nltk.corpus import stopwords
import pickle
import os
from nltk.corpus import wordnet

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('wordnet')

# Set environment variable to avoid certain errors
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"   

# Initialize stopwords set
stopwords_set = set(stopwords.words('english'))

def antonym(word):
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            if lemma.antonyms():
                return lemma.antonyms()[0].name()
    return word

def Negation(sentence):
    words = sentence.split()
    for i in range(len(words) - 1):
        if words[i] == "not" and i + 1 < len(words):
            antonym_word = antonym(words[i + 1])
            if antonym_word is not None:  # Check for None before assigning
                words[i + 1] = antonym_word
    return " ".join(words)  # Join words back into a sentence

def preprocess_data(text):
    # Lowercase the text
    text = text.lower()
    
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    
    # Remove stopwords
    stopwords_set = set(stopwords.words('english'))  # Ensure this is a set of stopwords
    text = ' '.join(word for word in text.split() if word not in stopwords_set)
    
    return text

max_text_length = 200

# Load the pretrained model
model = load_model('SentimentPatch1.keras')

# Load the previously saved tokenizer
with open('tokenizer.pkl', 'rb') as file:
    tokenizer = pickle.load(file)

# Summarize model structure
model.summary()

sample_texts = ["not bad movie at all"]
sample_texts[0] = Negation(sample_texts[0])

# Preprocess the text
sample_texts_preprocessed = preprocess_data(sample_texts[0])

# Convert text to sequences
sample_seqs = tokenizer.texts_to_sequences(sample_texts_preprocessed)

# Pad the sequences to ensure uniform length
sample_padded = pad_sequences(sample_seqs, maxlen=max_text_length, padding='post')

# Make predictions
predictions = model.predict(sample_padded)

# Output the sentiment scores
for i, prediction in enumerate(predictions):
    print(f"Text: {sample_texts[i]} - before preprocessing")
    print(f"Text: {sample_texts_preprocessed} - after preprocessing")
    
    print(f"Sentiment score: {prediction[0]}")

