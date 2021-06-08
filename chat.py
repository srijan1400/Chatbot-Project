import streamlit as st
import pickle
import numpy as np
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
import json

with open("intents.json") as file:
    data = json.load(file)

max_len = 20


def main():
    
    model = keras.models.load_model('chat_model')

    with open('chat_tokenizer.pkl', 'rb') as handle:
        tokenizer = pickle.load(handle)

    with open('chat_encoder.pkl', 'rb') as enc:
        label_enc = pickle.load(enc)


    st.title(" Deep Learning Chatbot")

    st.header("Made by Srijan Devnath")

    input = st.text_input("User :" " Ask a question")
    
    result = model.predict(keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences([input]),truncating='post', maxlen=max_len))

    tag  = label_enc.inverse_transform([np.argmax(result)])

    for i in data['intents']:
        if i['tag'] == tag:
            result2 = np.random.choice(i['responses'])
            st.text("Bot: ")
            st.success(result2)
            


if __name__ == "__main__":
    main()       

