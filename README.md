# !Sentiment intensity analyser
# Project Description
## The Sentiment Intensity Analyzer utilizes a Convolutional Neural Network (CNN) to classify text reviews as either positive or negative. The model is trained on the IMDb dataset, which includes a variety of movie reviews tagged  as positive or negative. The project preprocesses the text data, tokenizes it, and then uses a neural network model to predict the sentiment of new text inputs.
# Features
## -Text Preprocessing: Converts text to lowercase and removes punctuation.
## -Tokenization: Converts text into sequences of integers based on word frequency.
## -Padding: Ensures that all text sequences are of uniform length.
## -Model Architecture:
## -Embedding Layer
## -Convolutional Layer
## -Global Max Pooling Layer
## -Dense Layer
## -Dropout Layer
## -Output Layer with Sigmoid Activation
## Training and Validation: The model is trained for 10 epochs, with 20% of the data reserved for validation.
