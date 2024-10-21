import streamlit as st
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import pickle


nltk.download('stopwords')

#Loading model and vectorizer:

model_filename = 'pickled_final_model.pkl' 
with open(model_filename, 'rb') as file:
    model = pickle.load(file)

with open('pickled_vectorizer.pkl', 'rb') as vec_file:
    vectorizer = pickle.load(vec_file)

#Website 

st.title('Fake News Detection Application')
input_text = st.text_input('Enter the title and/or content of the news article:')

ps = PorterStemmer()
def stemming(text):
    stemmed_text = re.sub('[^a-zA-Z]'," ", text)
    stemmed_text = stemmed_text.lower()
    stemmed_text = stemmed_text.split()
    stemmed_text = [ps.stem(word) for word in stemmed_text if not word in stopwords.words('english')]
    stemmed_text = ' '.join(stemmed_text)
    print("data preproccessed")
    print(stemmed_text)
    return stemmed_text

def prediction(input_text):
    input_text = stemming(input_text)
    input_data = vectorizer.transform([input_text])
    prediction = model.predict(input_data)
    return prediction[0]

if input_text:
    pred = prediction(input_text)
    if pred == 1:
        st.write('The News is Fake!')
    else:
        st.write('The News is Real!')