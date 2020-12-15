import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from web.fake_news import predict
from web.fake_news.parser import PredictParser
from web.main import app
from web.utils.decorators import auth, body


@app.route('/', methods=['GET'])
def hello():
    return {'status': 'ok', 'payload': 'Hello!'}, 200


@app.route('/en/predict/', methods=['POST'])
@auth()
@body(PredictParser)
def predict_en(body):
    prediction = predict(
        body['content'],
        '/app/neural_net/train/english/output_english',
        PorterStemmer(),
        stopwords.words('english')
    )

    return {'status': 'ok', 'prediction': prediction}, 200


@app.route('/pt/predict/', methods=['POST'])
@auth()
@body(PredictParser)
def predict_pt(body):
    prediction = predict(
        body['content'],
        '/app/neural_net/train/portuguese/output_portuguese',
        nltk.RSLPStemmer(),
        stopwords.words('portuguese')
    )

    return {'status': 'ok', 'prediction': prediction}, 200
