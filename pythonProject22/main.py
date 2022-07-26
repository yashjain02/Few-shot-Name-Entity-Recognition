# Core Pkgs
import streamlit as st
import spacy_streamlit
import spacy
#nlp=spacy.load('en')
nlp = spacy.load("en_core_web_sm")

from bs4 import BeautifulSoup
from urllib.request import urlopen

import altair as alt
#import plotly.express as px

def get_text(raw_url):
    page=urlopen(raw_url)
    soup=BeautifulSoup(page)
    fetched_text=''.join(map(lambda p:p.text,soup.find_all))
    return fetched_text

def main():
    st.title("Named Entity Recognition APP ")
    menu = ["Home","NER","NER FOR URL"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        raw_text = st.text_area("Type Here")
        docx=nlp(raw_text)
        if st.button("Analyze"):
           spacy_streamlit.visualize_ner(docx,labels=nlp.get_pipe('ner').labels)

    elif choice =="NER":

        st.subheader("Named entity Recognition")

        with st.form(key='form'):
            raw_text = st.text_area("Type Here")
            product_list = []
            product_list.append(st.multiselect("Fruit", ["apple", "pear", "orange","banana","pear"]))
            product_list.append(st.multiselect("vegetables", ["broccoli", "spinach", "tomato","Onion","ginger"]))
            product_list.append(st.multiselect("Meat", ["chicken", "beef", "pork", "fish", "lamb"] ))
            submit_text = st.form_submit_button(label='Submit')
    elif choice=="NER FOR URL":
        st.subheader=("Analyze text from url")
        # if submit_text:
        #     col1, col2 = st.beta_columns(2)
        #     with col1:
        #         st.success("Original Text")
        #         st.write(raw_text)
        #
        #         st.success("Prediction")
        #
        #     with col2:
        #         st.success("Prediction Probability")





if __name__ == '__main__':
    main()
