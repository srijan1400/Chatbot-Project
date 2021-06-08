import json
import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import sklearn
from sklearn.preprocessing import LabelEncoder


with open('intents.json') as file:
    data = json.load(file)


training_sentences = []
training_labels = []
labels = []
responses = []


for intent in data['intents']:
    for pattern in intent['patterns']:
        training_sentences.append(pattern)
        training_labels.append(intent['tag'])
    responses.append(intent['responses'])


    if intent['tag'] not in labels:
        labels.append(intent['tag'])


num_classes = len(labels)

label_enc = LabelEncoder()

label_enc.fit(training_labels)

training_labels = label_enc.transform(training_labels)


vocab_size = 3000

embedding_dim = 16

max_len = 20

oov_token = "<OOV>"

tokenizer = Tokenizer(num_words= vocab_size, oov_token= oov_token)
tokenizer.fit_on_texts(training_sentences)
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(training_sentences)
padded_sequences = pad_sequences(sequences, truncating='post', maxlen=max_len)


model = Sequential()
model.add(Embedding(vocab_size, embedding_dim, input_length=max_len))
model.add(GlobalAveragePooling1D())
model.add(Dense(16, activation= 'relu'))
model.add(Dense(16, activation= 'relu'))
model.add(Dense(num_classes, activation='softmax'))


model.compile(loss = 'sparse_categorical_crossentropy', optimizer= 'adam', metrics= ['accuracy'])

model.summary()


history = model.fit(padded_sequences, np.array(training_labels), epochs= 1500)


model.save("chat_model")

with open('chat_tokenizer.pkl', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)


with open('chat_encoder.pkl', 'wb') as ecn_file:
    pickle.dump(label_enc, ecn_file,  protocol=pickle.HIGHEST_PROTOCOL)





