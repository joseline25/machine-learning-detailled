import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Load the dataset
data = pd.read_csv('reviews.csv')

# rename the features

data = data.rename(columns={'Unnamed: 0': 'unknown',
                   'User_name': 'user_name', 'Review title': 'review_title',
                            'Review Rating': 'review_rating', 'Review date': 'review_date', 'Review_body': 'review_body'})

# Initialize the SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()

# Perform sentiment analysis
sentiment_scores = []
for text in data['review_body']:
    scores = sia.polarity_scores(text)
    sentiment_scores.append(scores['compound'])

# Add sentiment scores to the dataset
data['sentiment_Score'] = sentiment_scores

# Define a function to recommend items based on sentiment scores
def recommend_items(user_threshold):
    recommended_items = data[data['sentiment_Score'] >= user_threshold]
    return recommended_items

# Get user's preferred sentiment threshold
user_threshold = 0.2  # Adjust this threshold according to user preferences

# Get recommended items based on user's preferred sentiment threshold
recommended_items = recommend_items(user_threshold)

# Print the recommended items
print(recommended_items)


"""
 Items with sentiment scores higher than or equal to this threshold will 
 be recommended to the user. You can adjust the threshold value according
 to your needs and experiment with different values to find the right balance
 between positive sentiment and personalized recommendations.
"""