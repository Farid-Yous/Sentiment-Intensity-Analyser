# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 11:15:21 2024

@author: farid
"""
import numpy as np
import matplotlib.pyplot as plt
import nltk
from datasets import load_dataset
import re
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout
import os
import pickle


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  

nltk.download('stopwords')
nltk.download('punkt')
dataset = load_dataset("imdb", download_mode="force_redownload")


def preprocess_data(text):
    # Convert text to lowercase
    text = text.lower()
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    
    return text
max_text_length = 200


train_dataset = dataset['train']
test_dataset = dataset['test']

#Preprocess all the data
train_dataset = train_dataset.map(lambda example: {'text': preprocess_data(example['text'])})
test_dataset = test_dataset.map(lambda example: {'text': preprocess_data(example['text'])})



max_vocab_size = 10000  # Maximum number of words in the vocabulary
tokenizer = Tokenizer(num_words=max_vocab_size, oov_token="<OOV>")

# Fit the tokenizer on the training data
tokenizer.fit_on_texts(train_dataset['text'])

# Convert the text to sequences of integers
X_train_seq = tokenizer.texts_to_sequences(train_dataset['text'])
X_test_seq = tokenizer.texts_to_sequences(test_dataset['text'])

# Pad the sequences to a uniform length
X_train_padded = pad_sequences(X_train_seq, maxlen=max_text_length, padding='post', truncating='post')
X_test_padded = pad_sequences(X_test_seq, maxlen=max_text_length, padding='post', truncating='post')

y_train = np.array(train_dataset['label'])
y_test = np.array(test_dataset['label'])

embedding_dim = 100

model = Sequential([
    # Embedding layer to convert word indices into dense vectors of fixed size
    Embedding(input_dim=max_vocab_size, output_dim=embedding_dim, input_length=max_text_length),
    
    # 1D convolutional layer
    Conv1D(filters=128, kernel_size=5, activation='relu'),
    
    # Global max pooling layer
    GlobalMaxPooling1D(),
    
    # Dense layer with ReLU activation
    Dense(64, activation='relu'),
    
    # Dropout layer for regularization
    Dropout(0.5),
    
    # Output layer with sigmoid activation for binary classification
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(
    X_train_padded,
    y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.2,  # Reserve 20% of training data for validation
    verbose=2
)
test_loss, test_accuracy = model.evaluate(X_test_padded, y_test, verbose=2)
print(f'Test Accuracy: {test_accuracy}')
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()
model.save('Sentiment.keras')

# Save the tokenizer
with open('tokenizer.pkl', 'wb') as file:
    pickle.dump(tokenizer, file)
    
sample_text = "horrible movie, i hated the experience"
sample_text_preprocessed = preprocess_data(sample_text)
sample_seq = tokenizer.texts_to_sequences([sample_text_preprocessed])
sample_padded = pad_sequences(sample_seq, maxlen=max_text_length, padding='post')
prediction = model.predict(sample_padded)
print(f"Sentiment score: {prediction[0][0]}")
