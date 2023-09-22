from tqdm.notebook import tqdm
from nltk.sentiment import SentimentIntensityAnalyzer
import warnings
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('tagsets')
nltk.download('words')
nltk.download('vader_lexicon')
nltk.help.upenn_tagset()


warnings.filterwarnings('ignore')

# set the style for our plotting
plt.style.use('ggplot')

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
# Quick EDA

# - get the count of value in review_rating column

print(data['review_rating'].value_counts())

"""
review_rating
10    257
9     240
8     199
7     153
6     102
1      69
5      63
4      36
3      36
2      30
Name: count, dtype: int64

"""

# trions le résultat précédent

print(data['review_rating'].value_counts().sort_index())

"""
review_rating
1      69
2      30
3      36
4      36
5      63
6     102
7     153
8     199
9     240
10    257
Name: count, dtype: int64

       
"""

# plot the previous displaying

data['review_rating'].value_counts().sort_index().plot(
    kind='bar', figsize=(10, 5), title="Count of reviews ratings")
plt.show()

# we have a huge rate of positive ratings, so the dataset is somehow biased


# BASIC NLTK

# pick one example and review (pick a random entry)

example = data['review_body'][27]
print(example)

"""
I mean, I'd say I liked The Hunger Games better - but not The Maze Runner. There's a real grimness to The Squid Games. Like The Platform. With every episode you're ground d
own deeper just like these poor saps who find themselves there. I liked it well enough though it was long and the characters pretty familiar to this genre. It does makes ma
ny apt comments about contemporary society - Ali's story is particularly sad as is Sae- Byeok's and the way we treat people just seeking a better life. I can't say I get th
e uproar of the series - Netflix's potential biggest- but it's worth a watch if you like dystopian stuff - and who doesn't?

                    28 out of 46 found this helpful.

                            Was this review helpful?  Sign in to vote.

"""

"""
à la fin de toutes les reviews on a le message 

  Was this review helpful?  Sign in to vote.
"""

# let's see some of the things nltk can do
# tokenize the sentence : split it into each word of the sentence
# the goal is to convert the text into some format that the computer can interpret

print(nltk.word_tokenize(example))


tokens = nltk.word_tokenize(example)

"""
['I', 'mean', ',', 'I', "'d", 'say', 'I', 'liked', 'The', 'Hunger', 'Games', 'better', '-', 'but', 'not', 'The', 'Maze', 'Runner', '.', 'There', "'s", 'a', 'real', 'grimnes
s', 'to', 'The', 'Squid', 'Games', '.', 'Like', 'The', 'Platform', '.', 'With', 'every', 'episode', 'you', "'re", 'ground', 'down', 'deeper', 'just', 'like', 'these', 'poor
', 'saps', 'who', 'find', 'themselves', 'there', '.', 'I', 'liked', 'it', 'well', 'enough', 'though', 'it', 'was', 'long', 'and', 'the', 'characters', 'pretty', 'familiar',
 'to', 'this', 'genre', '.', 'It', 'does', 'makes', 'many', 'apt', 'comments', 'about', 'contemporary', 'society', '-', 'Ali', "'s", 'story', 'is', 'particularly', 'sad', '
as', 'is', 'Sae-', 'Byeok', "'s", 'and', 'the', 'way', 'we', 'treat', 'people', 'just', 'seeking', 'a', 'better', 'life', '.', 'I', 'ca', "n't", 'say', 'I', 'get', 'the', '
uproar', 'of', 'the', 'series', '-', 'Netflix', "'s", 'potential', 'biggest-', 'but', 'it', "'s", 'worth', 'a', 'watch', 'if', 'you', 'like', 'dystopian', 'stuff', '-', 'an
d', 'who', 'does', "n't", '?', '28', 'out', 'of', '46', 'found', 'this', 'helpful', '.', 'Was', 'this', 'review', 'helpful', '?', 'Sign', 'in', 'to', 'vote', '.', 'Permalin
k']

"""
# find the part of speech for each token ie quand on a enlevé la ponctuation et créé
# une correspondance  en tuple
# example: ('mean', 'VBP') avec le mot mean et l'abbreviation
# VBP (Verb, non-3rd person singular present)
# qui est une forme de verbe
# https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html

print(nltk.pos_tag(tokens))

"""
[('I', 'PRP'), ('mean', 'VBP'), (',', ','), ('I', 'PRP'), ("'d", 'MD'), ('say', 'VB'), ('I', 'PRP'), ('liked', 'VBD'), ('The', 'DT'), ('Hunger', 'NNP'), ('Games', 'NNPS'),
('better', 'RBR'), ('-', ':'), ('but', 'CC'), ('not', 'RB'), ('The', 'DT'), ('Maze', 'NNP'), ('Runner', 'NNP'), ('.', '.'), ('There', 'EX'), ("'s", 'VBZ'), ('a', 'DT'), ('r
eal', 'JJ'), ('grimness', 'NN'), ('to', 'TO'), ('The', 'DT'), ('Squid', 'NNP'), ('Games', 'NNPS'), ('.', '.'), ('Like', 'IN'), ('The', 'DT'), ('Platform', 'NNP'), ('.', '.'
), ('With', 'IN'), ('every', 'DT'), ('episode', 'NN'), ('you', 'PRP'), ("'re", 'VBP'), ('ground', 'VBG'), ('down', 'RP'), ('deeper', 'RB'), ('just', 'RB'), ('like', 'IN'),
('these', 'DT'), ('poor', 'JJ'), ('saps', 'NNS'), ('who', 'WP'), ('find', 'VBP'), ('themselves', 'PRP'), ('there', 'RB'), ('.', '.'), ('I', 'PRP'), ('liked', 'VBD'), ('it',
 'PRP'), ('well', 'RB'), ('enough', 'RB'), ('though', 'IN'), ('it', 'PRP'), ('was', 'VBD'), ('long', 'RB'), ('and', 'CC'), ('the', 'DT'), ('characters', 'NNS'), ('pretty',
'RB'), ('familiar', 'JJ'), ('to', 'TO'), ('this', 'DT'), ('genre', 'NN'), ('.', '.'), ('It', 'PRP'), ('does', 'VBZ'), ('makes', 'VBZ'), ('many', 'JJ'), ('apt', 'JJ'), ('com
ments', 'NNS'), ('about', 'IN'), ('contemporary', 'JJ'), ('society', 'NN'), ('-', ':'), ('Ali', 'NNP'), ("'s", 'POS'), ('story', 'NN'), ('is', 'VBZ'), ('particularly', 'RB'
), ('sad', 'JJ'), ('as', 'IN'), ('is', 'VBZ'), ('Sae-', 'NNP'), ('Byeok', 'NNP'), ("'s", 'POS'), ('and', 'CC'), ('the', 'DT'), ('way', 'NN'), ('we', 'PRP'), ('treat', 'VBP'
), ('people', 'NNS'), ('just', 'RB'), ('seeking', 'VBG'), ('a', 'DT'), ('better', 'JJR'), ('life', 'NN'), ('.', '.'), ('I', 'PRP'), ('ca', 'MD'), ("n't", 'RB'), ('say', 'VB
'), ('I', 'PRP'), ('get', 'VBP'), ('the', 'DT'), ('uproar', 'NN'), ('of', 'IN'), ('the', 'DT'), ('series', 'NN'), ('-', ':'), ('Netflix', 'NNP'), ("'s", 'POS'), ('potential
', 'JJ'), ('biggest-', 'NN'), ('but', 'CC'), ('it', 'PRP'), ("'s", 'VBZ'), ('worth', 'IN'), ('a', 'DT'), ('watch', 'NN'), ('if', 'IN'), ('you', 'PRP'), ('like', 'VBP'), ('d
ystopian', 'JJ'), ('stuff', 'NN'), ('-', ':'), ('and', 'CC'), ('who', 'WP'), ('does', 'VBZ'), ("n't", 'RB'), ('?', '.'), ('28', 'CD'), ('out', 'IN'), ('of', 'IN'), ('46', '
CD'), ('found', 'VBD'), ('this', 'DT'), ('helpful', 'NN'), ('.', '.'), ('Was', 'NNP'), ('this', 'DT'), ('review', 'NN'), ('helpful', 'NN'), ('?', '.'), ('Sign', 'NNP'), ('i
n', 'IN'), ('to', 'TO'), ('vote', 'VB'), ('.', '.'), ('Permalink', 'VB')]

"""
tagged = nltk.pos_tag(tokens)

# put the tagged sentences into entity
entities = nltk.chunk.ne_chunk(tagged)
print(entities.pprint())

# STEP 1: VADER Sentiment analysis


sia = SentimentIntensityAnalyzer()

print(sia.polarity_scores("I'm so happy!"))
# {'neg': 0.0, 'neu': 0.318, 'pos': 0.682, 'compound': 0.6468}

print(sia.polarity_scores("This is the worst thing ever."))
# {'neg': 0.451, 'neu': 0.549, 'pos': 0.0, 'compound': -0.6249}

print(sia.polarity_scores(example))
# {'neg': 0.077, 'neu': 0.654, 'pos': 0.269, 'compound': 0.9844}
print("here")

# Run the polarity score on the entire dataset
res = {}
for i, row in data.iterrows():
    # print(row['review_rating'])
    text = row['review_body']
    myid = row['unknown']
    res[myid] = sia.polarity_scores(text)


vaders = pd.DataFrame(res).T

vaders = vaders.reset_index().rename(columns={'unknown': 'Id'})

merged_data = pd.concat([vaders, data], axis=1)
print(merged_data)

# plot Vader Score
ax = sns.barplot(data=merged_data, x='review_rating', y='compound')
ax.set_title('Compound Score for Review')
plt.show()


fig, axs = plt.subplots(1, 3, figsize=(12, 3))
sns.barplot(data=merged_data, x='review_rating', y='pos', ax=axs[0])
sns.barplot(data=merged_data, x='review_rating', y='neu', ax=axs[1])
sns.barplot(data=merged_data, x='review_rating', y='neg', ax=axs[2])
axs[0].set_title('Positive')
axs[1].set_title('Neutral')
axs[2].set_title('Negative')
plt.tight_layout()
plt.show()


# Step 3. Roberta Pretrained Model

from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from scipy.special import softmax

MODEL = f"cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)


print(example)
sia.polarity_scores(example)


# Run for Roberta Model

encoded_text = tokenizer(example, return_tensors='pt')
output = model(**encoded_text)
scores = output[0][0].detach().numpy()
scores = softmax(scores)
scores_dict = {
    'roberta_neg' : scores[0],
    'roberta_neu' : scores[1],
    'roberta_pos' : scores[2]
}
print(scores_dict)

