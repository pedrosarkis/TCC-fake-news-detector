import re
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import one_hot
import numpy as np
from web.main import vocabulary_size, neural_input_length, model


def predict(content, neural_net_data_path, stemmer, stopwords):
    corpus = []
    review = re.sub('[^a-zA-Z]', ' ', content)
    review = review.lower()
    review = review.split()
    review = [stemmer.stem(word) for word in review if word not in stopwords]
    review = ' '.join(review)
    corpus.append(review)

    bag_of_words = [one_hot(words, vocabulary_size) for words in corpus]
    X = np.array(pad_sequences(bag_of_words, padding='pre', maxlen=neural_input_length))

    model.load_weights(neural_net_data_path)
    prediction = model.predict(X).tolist()[0][0]
    return prediction