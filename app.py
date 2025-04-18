import streamlit as st
import pickle
from string import punctuation
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

#Defining a function to preprocess text
def transform_text(text):
    punc = list(punctuation)
    stop = stopwords.words('english')
    bad_tokens = stop + punc
    tokens = word_tokenize(text)
    lemma = WordNetLemmatizer()
    word_tokens = [t for t in tokens if t.isalpha()]
    clean_tokens = [lemma.lemmatize(t.lower()) for t in word_tokens if t not in bad_tokens]
    return ' '.join(clean_tokens)

#loading the model and the vectorizer
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

#Creating the user interface
st.title('Email Spam Classifier')
input_field = st.text_area('Input here')


if st.button("Predict"):

    #Creating the pipeline for the model
    transformed_text = transform_text(input_field)
    vectorized_text = vectorizer.transform([transformed_text])
    result = model.predict(vectorized_text)[0]

    #Display the output
    if result == 1:
        st.header('Spam')
    else:
        st.header("Not Spam")