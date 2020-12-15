# Estratégia:

# 1- Leitura e pré-processamento dos dados
# Justificativa: Obter o vetor X (dados input da rede) e vetor y (classes da rede)

# 2- Aplicação do algoritmo de Bag of Words (one-hot representation)
# Justificativa: Obter uma classificação numérica das frases (quantizar a informação)

# 3- Configuração da rede neural
# Justificativa: Definição do algoritmo, composto por 5 camadas:
# 1) Embedding: Representação do texto em features
# 2) Dropout: Desliga parte da rede aleatóriamente para evitar overfitting (decoramento dos valores)
# 3) LSTM: É um tipo de neurônio que possui memória, e é a principal
#   abordagem para redes neurais do tipo recorrente, uma das melhores abordagens atualmente,
#   e que evitam problemas de vanish gradient (situações em que a rede aprende de maneira ineficiente)
# 4) Dropout: ...
# 5) Dense: Reduz o número de neurônios ao número de classes

# 4- Treinamento da rede neural
# Justificativa: Geração da rede neural de detecção de fake news

# 5- Apresentação dos resultados
# Justificativa: Demonstra acurácia, falsos positivos e falsos negativos


######
# 1- Leitura e pré-processamento dos dados


# Biblioteca voltada para o processamento de dados na memória
import pandas as pd

# id, title, author, text, label
# df = dataframe
df = pd.read_csv('train/english/input_english.csv')

# Limpa qualquer linha que possua alguma coluna nula
df = df.dropna()

# Exclui a coluna (axis 1) que contém as labels
X = df.drop('label', axis=1)

# Captura os rótulos de cada notícia (0: notícia verdadeira, 1: notícia falsa)
y = df['label']

# Reorganiza índices (necessário se houve deleção de dados)
X.reset_index(inplace=True)


######
# 2- Aplicação do algoritmo de Bag of Words (one-hot representation)


# Um Bag of Words é um vetor que representa uma frase
# Cada palavra possui um vetor esparso identificador composto de 0 e 1
#  Ex: eu gosto do que eu aprendo
#      eu como banana
#    tamanho do vocabulário = 7
#    eu       = [0, 0, 0, 0, 0, 0, 1]
#    gosto    = [0, 0, 0, 0, 0, 1, 0]
#    do       = [0, 0, 0, 0, 1, 0, 0]
#    que      = [0, 0, 0, 1, 0, 0, 0]
#    aprendo  = [0, 0, 1, 0, 0, 0, 0]
#    como     = [0, 1, 0, 0, 0, 0, 0]
#    banana   = [1, 0, 0, 0, 0, 0, 0]
# A representação da frase seria:
#    eu gosto do que eu aprendo: [0, 0, 1, 1, 1, 1, 2]
#    eu como banana            : [1, 1, 0, 0, 0, 0, 1]

# Natural Language ToolKit: Nos ajudara com processamento de linguagem natural (NLP)
import nltk

# Para diminuir o tamanho dos vetores de bag of words:

# a) Vamos remover palavras de pouco significado semântico (Stop Words)
# ex: as, e, os, de, para, com, sem, foi
from nltk.corpus import stopwords
nltk.download('stopwords')

# b) Também, vamos pegar apenas a stemming word (palavra raiz) da original
# ex: boca, bocadas = boc
from nltk.stem.porter import PorterStemmer

import re
ps = PorterStemmer()
corpus = []
for i in range(0, len(X)):
    # Remove tudo que não for letra
    review = re.sub('[^a-zA-Z]', ' ', X['title'][i] + ' ' + X['text'][i])
    # Torna tudo minúsculo
    review = review.lower()
    # Pega cada palavra separadas por espaços e forma um array
    review = review.split()

    # Aplica o algoritmo de stemming
    review = [ps.stem(word) for word in review if word not in stopwords.words('english')]  # TODO: português
    # Reforma a frase separando cada palavra por espaços
    review = ' '.join(review)
    # Adiciona à lista de frases
    corpus.append(review)

    print(f'Processing {i} of {len(X)}')


# Total de palavras estimadas
vocabulary_size = 5000

# Aplica o algoritmo de bag of words, só que ao invés de vetores esparsos gera identificadores únicos para cada palavra
# Obs: Cada identificador é gerado a partir de um algoritmo de hash, então pode haver colisões
from tensorflow.keras.preprocessing.text import one_hot
bag_of_words = [one_hot(words, vocabulary_size) for words in corpus]

# Deixa os vetores iguais em tamanho, adicionando zeros à esquerda
# ex (maxlen = 4): {[2, 1, 4, 3], [3, 1, 2]} = {[2, 1, 4, 3], [0, 3, 1, 2]}
from tensorflow.keras.preprocessing.sequence import pad_sequences
# Serão adicionados zeros até o vetor possuir 20 valores
neural_input_length = 20
input_net_data = pad_sequences(bag_of_words, padding='pre', maxlen=neural_input_length)


######
# 3- Configuração da rede neural

# Permite criar uma rede neural composta de layers sequenciais
from tensorflow.keras.models import Sequential
model = Sequential()

# Algoritmo de word embedding
# Vetores de bag of words costumam ser muito grandes, além de serem pouco representativos
# Foi então criado algoritmos de Word Embedding (GloVe, Word2Vec), onde palavras são representadas
#  por vetores de características n dimensionais, cada dimensão representando uma característica (ex: gênero, sentimento)
# Através dessas features, é possível aplicar algebra linear (distância, cosseno de similaridade) e obter informações mais relevantes,
#  além de ser representável com bem menos dados e não ser esparso
from tensorflow.keras.layers import Embedding

# Total de dimensões do vetor (features) a serem descobertas
# TODO: Ao invés de treinar a rede para descobrir as features, usar a rede já treinada do word2vec ou glove
embedding_vector_features = 40
model.add(Embedding(vocabulary_size, embedding_vector_features, input_length=neural_input_length))

# Desliga 30% da rede
from tensorflow.keras.layers import Dropout
model.add(Dropout(0.3))

# Permite criar um layer com 100 units (neurônios complexos) LSTM (Long Short Term Memory)
from tensorflow.keras.layers import LSTM
model.add(LSTM(100))
model.add(Dropout(0.3))

# Afunila o final da rede para possuir o número de neurônios equivalente ao número de classes
# Observe que o resultado é um neurônio, onde x < 0.5 é fake news, senão confiável
from tensorflow.keras.layers import Dense
model.add(Dense(1, activation='sigmoid'))

# Executa as otiimizações para finalmente montar a rede
# binary_crossentropy: Rede otimizada para classificação binária
# adam: Otimiza utilizando descida do gradiente
# accuracy: Exibe a precisão enquanto estiver treinando
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Observe a complexidade da rede a ser treinada (mais de 25 mil operações por época)
print(model.summary())


######
# 4- Treinamento da rede neural

# Prepara os dados de entrada e saída do treinamento
import numpy as np
X = np.array(input_net_data)
y = np.array(y)
print(X.shape, y.shape)

# Configuração de holdout: Define quantos porcento da base será utilizada para testar a rede
# e qual é o embaralhamento
# stratify: força pegar na mesma proporção as duas classes para teste
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42, stratify=y)

# Executa o treinamento
# epochs: Quantas vezes ele vai executar o treinamento
# batch_size: De quantas em quantas instâncias vai puxar na memória por vez
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=64)
model.save_weights('train/english/output_english')

######
# 5- Apresentação dos resultados

# Roda o classificador e coleta os resultados
y_pred = model.predict_classes(X_test)

# Monta a matriz de confusão (matriz que demonstra acertos, falsos positivos e falsos negativos)
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, y_pred))

# Exibe precisão da classificação
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred))
