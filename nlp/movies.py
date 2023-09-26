import nltk
import random
from nltk.corpus import movie_reviews

# Prepare the dataset
nltk.download('movie_reviews')

# Create a list of documents, each consisting of a list of words and its sentiment label
documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

# documents break into words

# Shuffle the documents
random.shuffle(documents)

# Define the feature extractor
def document_features(document):
    words = set(document)
    features = {}
    for word in words:
        features['contains({})'.format(word)] = (word in words)
        return features

# Extract the feature sets from the documents
featuresets = [(document_features(d), c) for (d, c) in documents]

# Split the dataset into training and testing sets
train_set, test_set = featuresets[:1500], featuresets[1500:]

# Train the classifier
classifier = nltk.NaiveBayesClassifier.train(train_set)

# Evaluate the classifier
accuracy = nltk.classify.accuracy(classifier, test_set)
print('Classifier accuracy:', accuracy) # environ 50% chaque fois

# Classify new text
text = "I  did not like the movie!"
features = document_features(text.split())
sentiment = classifier.classify(features)
print('Sentiment:', sentiment)