import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Load the dataset
data = pd.read_csv('reviews.csv')

print(data.info())

"""
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 1185 entries, 0 to 1184
Data columns (total 6 columns):
 #   Column         Non-Null Count  Dtype
---  ------         --------------  -----
 0   Unnamed: 0     1185 non-null   int64
 1   User_name      1185 non-null   object
 2   Review title   1185 non-null   object
 3   Review Rating  1185 non-null   object
 4   Review date    1185 non-null   object
 5   Review_body    1185 non-null   object
dtypes: int64(1), object(5)
memory usage: 55.7+ KB

"""

print(data.columns)

"""
['Unnamed: 0', 'User_name', 'Review title', 'Review Rating',
       'Review date', 'Review_body']
"""

# rename the features

data = data.rename(columns={'Unnamed: 0': 'unknown',
                   'User_name': 'user_name', 'Review title': 'review_title',
                            'Review Rating': 'review_rating', 'Review date': 'review_date', 'Review_body': 'review_body'})

print(data.info())

"""
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 1185 entries, 0 to 1184
Data columns (total 6 columns):
 #   Column         Non-Null Count  Dtype
---  ------         --------------  -----
 0   unknown        1185 non-null   int64
 1   user_name      1185 non-null   object
 2   review_title   1185 non-null   object
 3   review_rating  1185 non-null   object
 4   review_date    1185 non-null   object
 5   review_body    1185 non-null   object
dtypes: int64(1), object(5)
memory usage: 55.7+ KB

"""

# Preprocessing
# Perform any necessary preprocessing steps on the text data in the "Review_body" column

# Split the data into features and target
X = data['review_body']

# transform the text in rating as integer (from 1 to 10)
vals = []
for i in data['review_rating']:
    vals.append(int(i[:i.find('/')]))

data['review_rating'] = vals
y = data['review_rating']
print(data)


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer()

# Fit and transform the training data
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)

# Transform the testing data
X_test_tfidf = tfidf_vectorizer.transform(X_test)

print("**************************Logistic Regression*************************************")

# Initialize the logistic regression model
logistic_model = LogisticRegression()

# Train the logistic regression model
logistic_model.fit(X_train_tfidf, y_train)

# Predict the sentiment labels for the testing data
y_pred = logistic_model.predict(X_test_tfidf)

# Calculate the accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy) # 25,31%


"""

To build a Natural Language Processing (NLP) model for sentiment prediction based on 
the provided dataset, you can follow these steps:

1. **Data Preprocessing:** Preprocess the text data in the "Review_body" 
column to make it suitable for NLP tasks. This may involve steps such as removing 
punctuation, converting text to lowercase, removing stop words, and performing 
stemming or lemmatization.

2. **Feature Extraction:** Convert the preprocessed text data into numerical
features that can be used by a machine learning model. Common techniques for 
feature extraction in NLP include bag-of-words, TF-IDF (Term Frequency-Inverse
Document Frequency), or word embeddings like Word2Vec or GloVe.

3. **Data Split:** Split the dataset into training and testing sets to
train and evaluate the model's performance. Typically, a split of around 70-80%
for training and 20-30% for testing is used.

4. **Model Selection:** Choose a suitable machine learning algorithm for sentiment 
prediction. Common choices include Logistic Regression, Naive Bayes, Support Vector
Machines, or even more advanced models like Recurrent Neural Networks (RNNs) or 
Transformers.

5. **Model Training:** Train the selected model using the training data. Fit the model
to the extracted features and the corresponding sentiment labels.

6. **Model Evaluation:** Evaluate the trained model's performance on the testing set 
using appropriate evaluation metrics such as accuracy, precision, recall, or F1 score.

7. **Model Fine-tuning:** Depending on the performance of the initial model, you can 
fine-tune hyperparameters or experiment with different model architectures to improve performance. Techniques like grid search or randomized search can help in finding optimal hyperparameter settings.

8. **Prediction:** Once you are satisfied with the model's performance, you can use 
it to make predictions on new, unseen data.

It's important to note that the specific implementation details and choice of algorithms
may vary based on the size of the dataset, the complexity of the sentiment patterns, and
other factors. Additionally, you may also consider techniques like data augmentation, 
handling class imbalance, or using pre-trained language models to further enhance the 
model's performance.

Remember to adapt and customize the steps according to your specific requirements 
and dataset for sentiment prediction in NLP.

breakdown of the code

First, the dataset is loaded using pd.read_csv().

Next, you can perform any necessary preprocessing steps on the "Review_body"
column to clean and prepare the text data.

The dataset is split into the feature column (X) and the target column (y).

The dataset is further split into training and testing sets using train_test_split().

A TF-IDF vectorizer is initialized using TfidfVectorizer().

The training data is fit and transformed using fit_transform() to convert
the text data into numerical features.

The testing data is transformed using transform() based on the fitted 
TF-IDF vectorizer.

A logistic regression model is initialized using LogisticRegression().

The logistic regression model is trained using fit() with the transformed
training data and corresponding sentiment labels.

Sentiment labels are predicted for the testing data using predict().


The accuracy of the model is calculated using accuracy_score() by comparing 
the predicted labels (y_pred) with the actual labels (y_test).
"""

print("**************************LinearSVC*************************************")

from sklearn.svm import LinearSVC



# Initialize the TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer()

# Fit and transform the training data
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)

# Transform the testing data
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Initialize the linear SVM classifier
svm_classifier = LinearSVC()

# Train the classifier
svm_classifier.fit(X_train_tfidf, y_train)

# Predict the sentiment labels for the testing data
y_pred = svm_classifier.predict(X_test_tfidf)

# Calculate the accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy) # 29,5%

print("**************************Using NLTK*************************************")



import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score



# Preprocessing
nltk.download('stopwords')
nltk.download('punkt')

stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # Tokenize the text
    tokens = word_tokenize(text)
    
    # Remove stopwords
    tokens = [token.lower() for token in tokens if token.lower() not in stop_words and token.isalpha()]
    
    # Join the tokens back into a single string
    preprocessed_text = ' '.join(tokens)
    
    return preprocessed_text

# Apply preprocessing to the text data
data['preprocessed_text'] = data['review_body'].apply(preprocess_text)

# Split the data into features and target
X = data['preprocessed_text']
y = data['review_rating']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer()

# Fit and transform the training data
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)

# Transform the testing data
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Initialize the Naive Bayes classifier
naive_bayes = MultinomialNB()

# Train the classifier
naive_bayes.fit(X_train_tfidf, y_train)

# Predict the sentiment labels for the testing data
y_pred = naive_bayes.predict(X_test_tfidf)

# Calculate the accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)# 26,5


print("**************************RandomForest*************************************")

# Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train_tfidf, y_train)

# Predict using Random Forest Classifier
rf_y_pred = rf_classifier.predict(X_test_tfidf)

# Calculate accuracy for Random Forest Classifier
rf_accuracy = accuracy_score(y_test, rf_y_pred)
print("Random Forest Accuracy:", rf_accuracy) # 25,7%


print("**************************Support Vector Machine*************************************")

# Support Vector Machine (SVM) Classifier
svm_classifier = SVC(kernel='linear', random_state=42)
svm_classifier.fit(X_train_tfidf, y_train)

# Predict using SVM Classifier
svm_y_pred = svm_classifier.predict(X_test_tfidf)

# Calculate accuracy for SVM Classifier
svm_accuracy = accuracy_score(y_test, svm_y_pred)
print("SVM Accuracy:", svm_accuracy)

print("*************************SentimentIntensityAnalyzer***************************")

from nltk.sentiment import SentimentIntensityAnalyzer

# Initialize the SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()

# Perform sentiment analysis
sentiment_scores = []
for text in data['review_body']:
    scores = sia.polarity_scores(text)
    sentiment_scores.append(scores['compound'])

# Assign sentiment labels based on the compound scores
sentiment_labels = ['positive' if score >= 0.05 else 'negative' if score <= -0.05 else 'neutral' for score in sentiment_scores]

# Add sentiment labels to the dataset
data['review_rating'] = sentiment_labels

# Print the dataset with sentiment labels
print(data.T)
