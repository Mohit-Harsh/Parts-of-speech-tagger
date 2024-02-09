import streamlit as st

import pandas as pd
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pickle
from collections import defaultdict

with open('pos_tags.json','r') as file:
    tags = json.load(file)

with open('tokenizer.pickle','rb') as file:
    tokenizer = pickle.load(file)

model = keras.models.load_model('pos_predictor.keras')

if 'messages' not in st.session_state:
    st.session_state.messages=[]

for message in st.session_state.messages:
    with st.chat_message(message['role']):
        if message['role'] == 'user':
            st.write(message['content'])
        else:
            st.dataframe([message['content']])

def predict(query):

    d = defaultdict(str)

    query_seq = tokenizer.texts_to_sequences([query])
    query_seq_padded = keras.preprocessing.sequence.pad_sequences(query_seq, maxlen=271, padding='post',
                                                                  truncating='post')
    results = model.predict([query_seq_padded])

    for i in range(len(query_seq[0])):
        token = tokenizer.sequences_to_texts([[query_seq[0][i]]])
        tag = tags[str(np.argmax(results[-1][i]))]

        d[token[0]] = tag

    with st.chat_message('assistant'):

        table = st.dataframe(data=[d])
        st.session_state.messages.append({'role':'assistant','content':d})



query = st.chat_input()

if query:

    with st.chat_message('user'):
        st.write(query)
        st.session_state.messages.append({'role':'user','content':query})

    predict(query)