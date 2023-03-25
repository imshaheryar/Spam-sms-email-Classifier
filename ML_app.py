import streamlit as st
import pickle 
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
p = PorterStemmer()

tv = pickle.load(open('vectorizer', 'rb'))
# df = pickle.load(open('df.pkl', 'rb'))
model = pickle.load(open('BornoulliNB.pkl', 'rb'))



def trans_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text) #lowercase
    x = []
    for i in text: #removing special characters
        if i.isalnum():            #aphanumeric
            x.append(i)
    y = []
    for i in x: #remove punctuations
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    z = []
    for i in y: #removing sentences like ing
        a = p.stem(i) 
        z.append(a)
    return ' '.join(z)

st.title('EMAIL/SMS SPAM CLASSIFIER')

sms = st.text_area('Enter the Message', help = 'Enter sms')

if st.button('Predict'):

    trans = trans_text(sms) # fuction applied for preprocesing of data

    vector = tv.transform([trans]) # vectorize

    result = model.predict(vector)[0] #prdict

    if result == 0:
        st.header('Not spam')
    else:
        st.header('Spam')