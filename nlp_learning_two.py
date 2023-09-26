import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer # tokenizer API

sentences = [
    'I love my dog',
    'I love my cat'
]

tokenizer = Tokenizer(num_words=100)  # max numbers of words to keep
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index

print(word_index)

