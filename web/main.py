import nltk
from flask import Flask
from flask_cors import CORS

from web.utils import DecimalEncoder

app = Flask(__name__)
app.json_encoder = DecimalEncoder
CORS(app)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense

embedding_vector_features = 40
vocabulary_size = 5000
neural_input_length = 20

nltk.download('stopwords')
nltk.download('rslp')

model = Sequential()
model.add(Embedding(vocabulary_size, embedding_vector_features, input_length=neural_input_length))
model.add(Dropout(0.3))
model.add(LSTM(100))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# noinspection PyUnresolvedReferences
import web.fake_news.api
