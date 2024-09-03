# Sentiment intensity analyser

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

# Training
![Before removal of stopwords](Before(stopwords).png)
![After removal of stopwords](After(stopwords).png)
## In patch 1, i decided to remove all stopwords (words such as a, the , and etc.) from training to optimise the accuracy of my model, this removes the weight on stopwords. As you can see from the cross validation performance, the removal of stopwords increased it drastically.
## Also in patch 1, I added a simple negation handler which I will be working more on in the future that turns words such as "not good" into words like "bad", this stops the model from reading the word "good" and giving it a positive score.
# Usage
## The trained model is stored in the .keras file and used in the model-testing.py file where it is imported and tested. Just alter the sample text and run the model-testing file where the output will be between 0(negative) and 1(positive)

