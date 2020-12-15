import pandas as pd
import nltk
from nltk.corpus import stopwords
import re
import numpy as np
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


df = pd.read_csv('train/portuguese/input_portuguese.csv')
df = df.dropna()
X = df.drop('label', axis=1)
y = df['label'].map(lambda x: 1 if x != 'fake' else 0).tolist()
X.reset_index(inplace=True)

nltk.download('stopwords')
nltk.download('rslp')
ps = nltk.RSLPStemmer()
corpus = []
for i in range(0, len(X)):
    review = re.sub('[^a-zA-Z]', ' ', 'os ' + X['text'][i])
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if word not in stopwords.words('portuguese')]
    review = ' '.join(review)
    corpus.append(review)

    print(f'Processing {i} of {len(X)}')

vocabulary_size = 5000
bag_of_words = [one_hot(words, vocabulary_size) for words in corpus]
neural_input_length = 20
input_net_data = pad_sequences(bag_of_words, padding='pre', maxlen=neural_input_length)

model = Sequential()
embedding_vector_features = 40
model.add(Embedding(vocabulary_size, embedding_vector_features, input_length=neural_input_length))
model.add(Dropout(0.3))
model.add(LSTM(100))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

X = np.array(input_net_data)
y = np.array(y)
print(X.shape, y.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=42, stratify=y)
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=11, batch_size=64)
model.save_weights('train/portuguese/output_portuguese')

y_pred = model.predict_classes(X_test)
print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test, y_pred))
