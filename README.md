# NLP
**Code Cell 1
import pandas as pd
data = pd.read_csv("/content/drive/MyDrive/IMDB Dataset.csv")
print(data.head())
**Explanation:**

Imports the pandas library.
Reads an IMDB dataset (IMDB Dataset.csv) from a given path into a DataFrame.
Displays the first five rows of the dataset.

**Code Cell 2**

import os
import io
import string
from tqdm import tqdm
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np

import nltk
from nltk.corpus import stopwords

from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import matplotlib.pyplot as plt
from wordcloud import WordCloud
**Explanation:**

Imports libraries for text preprocessing (string, BeautifulSoup, nltk), data visualization (matplotlib, WordCloud), machine learning (tensorflow, sklearn), and data handling (pandas, numpy).

**Code Cell 3**

text = " ".join(data['review'].astype(str))

# Generate the word cloud
wordcloud = WordCloud(
    width=800,
    height=400,
    background_color='white',
    colormap='plasma',
    max_words=200
).generate(text)

# Display the word cloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()
**Explanation:**

Joins all reviews into a single string for word cloud generation.
Creates a visual representation of the most frequent words in the dataset.

**Code Cell 4**

data['sentiment'] = data['sentiment'].map({'positive': 1, 'negative': 0})
**Explanation:**

Converts the sentiment column to binary values: positive → 1, negative → 0.

**Code Cell 5**

# Shuffle the DataFrame
shuffled_data = data.sample(frac=1, random_state=2023)

train_size = 0.8  # 80% for training, 20% for testing
train_data, test_data = train_test_split(shuffled_data, train_size=train_size, random_state=2023)
**Explanation:**

Randomizes the dataset rows.
Splits the data into training (80%) and testing (20%) sets.

**Code Cell 6**

print(f'Train shappe: {train_data.shape}')
print(f'Test shappe: {test_data.shape}')
**Explanation:**

Prints the shapes of the training and testing datasets.

**Code Cell 7**

X_train = train_data['review']
y_train = train_data['sentiment']

X_test = test_data['review']
y_test = test_data['sentiment']
**Explanation:**

Splits the dataset into input (X_train, X_test) and output (y_train, y_test) variables.

**Code Cell 8**

# using stopwords from nltk
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
**Explanation:**

Downloads and sets English stopwords for text preprocessing.

**Code Cell 9**

table = str.maketrans('', '', string.punctuation)
**Explanation:**

Creates a translation table to remove punctuation from text.

**Code Cell 10**

X_train_cleaned = []

for item in tqdm(X_train):
    sentence = str(item).lower()
    sentence = sentence.replace(",", " , ").replace(".", " . ")
    soup = BeautifulSoup(sentence)
    sentence = soup.get_text()
    words = sentence.split()
    filtered_sentence = " ".join(
        word.translate(table) for word in words if word not in stop_words
    )
    X_train_cleaned.append(filtered_sentence)
**Explanation:**

Preprocesses training text: lowercasing, removing HTML tags, punctuation, and stopwords.

**Code Cell 11-12**

sentence_lengths = [len(sentence) for sentence in X_train_cleaned]
plt.plot(sorted(sentence_lengths))
**Explanation:**

Computes and plots the length distribution of sentences.

**Code Cell 13**

all_words = [word for sentence in X_train_cleaned for word in sentence.split()]
total_vocab_size = len(set(all_words))
print("Total Vocabulary Size:", total_vocab_size)
**Explanation:**

Calculates and prints the vocabulary size from the training data.

**Code Cell 14-16**

vocab_size = 50000
max_length = 2000
tokenizer = Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(X_train_cleaned)
**Explanation:**

Configures a tokenizer for sequence preparation with a vocabulary size of 50,000.

**Code Cell 17-19**

training_padded, validation_padded, y_train, y_valid = train_test_split(training_padded, y_train)
**Explanation:**

Splits the training data into training and validation sets for model evaluation.

**Code Cell 20-21**

def create_model(vocab_size, embedding_dim):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(24, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
**Explanation:**

Defines a neural network model for sentiment classification.

**Code Cell 22-24**

model = create_model(vocab_size, embedding_dim)
model.fit(training_padded, y_train, validation_data=(validation_padded, y_valid))
**Explanation:**

Compiles and trains the model on the padded training data.

**Code Cell 25-28**

plot_history(history, "accuracy")
model.evaluate(testing_padded, testing_labels)
Explanation:

Plots the training/validation accuracy and evaluates the model on the test set.

** Cell 29**

reverse_word_index = {value: key for key, value in tokenizer.word_index.items()}
**Explanation:**

Saves word embeddings for visualization in tools like TensorBoard.
