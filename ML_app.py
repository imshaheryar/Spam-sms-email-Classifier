import streamlit as st
import pickle 
import string



tv = pickle.load(open('vectorizer', 'rb'))
tran = pickle.load(open('transform.pkl', 'rb'))
model = pickle.load(open('BornoulliNB.pkl', 'rb'))





st.title('EMAIL/SMS SPAM CLASSIFIER')

sms = st.text_area('Enter the Message', help = 'Enter sms')

if st.button('Predict'):

    trans = tran(sms) # fuction applied for preprocesing of data

    vector = tv.transform([trans]) # vectorize

    result = model.predict(vector)[0] #prdict

    if result == 0:
        st.header('Not spam')
    else:
        st.header('Spam')
