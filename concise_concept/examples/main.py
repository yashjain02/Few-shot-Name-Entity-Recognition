import streamlit as st
import spacy_streamlit as sst
import concise_concepts
import spacy
from bs4 import BeautifulSoup
from urllib.request import urlopen
import nltk
#nltk.download()
from spacy import displacy
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from bs4 import BeautifulSoup

HTML_WRAPPER = """<div style="overflow-x":auto;border:1px solid #e6e9ef;border-radius:0.25rem;padding: 1rem">{}</div>"""

from spacy import displacy

# nlp=spacy.load('en')
nlp = spacy.load("en_core_web_lg")

# from gensim.summarization import summarize
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
# from sumy.summarizers.lex_rank import LexRankSummarize
from bs4 import BeautifulSoup
from urllib.request import urlopen
data = {
    "fruit": ["apple", "pear", "orange"],
    "vegetable": ["broccoli", "spinach", "tomato"],
    "meat": ["chicken", "beef", "pork", "fish", "lamb"],
    "Product": ["Card"],
    "Activity": ['Payment']
}

def recipe(model):
    product_list = {}
    raw_text = st.text_area("Type Here")
    product_list['fruit'] = (st.multiselect("Fruit", ["apple", "pear", "orange"]))
    product_list['vegetable'] = st.multiselect("vegetables", ["broccoli", "spinach", "tomato", "Onion", "ginger"])
    product_list['meat'] = st.multiselect("Meat", ["chicken", "beef", "pork", "fish", "lamb"])
    print(product_list)
    if st.button('predict'):
        nlp = spacy.load(model, disable=["ner"])
        nlp.add_pipe("concise_concepts",
                     config={"data": product_list, "topn": [100, 100, 100], "ent_score": True})
        doc = nlp(raw_text)
        sst.visualize_ner(doc, show_table=False, )


def NER():
    domain = st.selectbox('domain', ['recipe', 'Banking'])
    product_list = {}
    if domain == "recipe":
        raw_text = st.text_area("Type Here")
        product_list['fruit'] = (st.multiselect("Fruit", ["apple", "pear", "orange"]))
        product_list['vegetable'] = st.multiselect("vegetables", ["broccoli", "spinach", "tomato", "Onion", "ginger"])
        product_list['meat'] = st.multiselect("Meat", ["chicken", "beef", "pork", "fish", "lamb"])
        print(product_list)
        if st.button('predict'):
            nlp = spacy.load('en_core_web_lg', disable=["ner"])
            nlp.add_pipe("concise_concepts",
                         config={"data": product_list, "topn": [100, 100, 100], "ent_score": True})
            doc = nlp(raw_text)
            sst.visualize_ner(doc, show_table=False, )
    elif domain == "Banking":
        raw_text = st.text_area("Type Here")
        product_list['activity'] = (st.multiselect("activity", ["Payment", "Open account", "past due"]))
        product_list['product'] = st.multiselect("product", ["Card", "credit card"])
        print(product_list)
        if st.button('predict'):
            nlp = spacy.load('en_core_web_lg', disable=['ner'])
            nlp.add_pipe("concise_concepts",
                         config={"data": product_list, "ent_score": True})
            doc = nlp(raw_text)
            sst.visualize_ner(doc, show_table=False, )
    elif domain == "Banking":
        raw_text = st.text_area("Type Here")
        product_list['activity'] = (st.multiselect("activity", ["Payment"]))
        product_list['product'] = st.multiselect("product", ["Card"])
        product_list['Money'] = st.multiselect("Money", ["dollars", "euros", "100 dollars", "100$"])
        print(product_list)
        if st.button('predict'):
            nlp = spacy.load('en_core_web_lg', disable=["ner"])
            nlp.add_pipe("concise_concepts",
                         config={"data": product_list, "ent_score": True})
            doc = nlp(raw_text)
            sst.visualize_ner(doc, show_table=False)


def Non_NER():
    raw_text = st.text_area("Type Here")
    model = st.selectbox('Choose the model', ['en_core_web_lg', 'en_core_web_md'])
    nlp = spacy.load(model)
    # nlp.add_pipe("concise_concepts",config={"data":data})
    doc = nlp(raw_text)
    sst.visualize_ner(doc, show_table=False)


def URL():
    nlp = spacy.load("en_core_web_lg")

    def get_text(raw_url):
        page = urlopen(raw_url)
        soup = BeautifulSoup(page)
        fetched_text = ''.join(map(lambda p: p.text, soup.find_all('p')))
        return fetched_text

    st.subheader = ("Analyze text from url")
    raw_url = st.text_input("enter URL", "Type here")
    if st.button("Extract"):
        if raw_url != "Type here":
            result = get_text(raw_url)
            doc = nlp(result)
            sst.visualize_ner(doc, show_table=False)


def Text_summary():
    def sumy_summarizer(docx):
        parser = PlaintextParser.from_string(docx, Tokenizer("english"))
        lex_summarizer = LexRankSummarizer()
        summary = lex_summarizer(parser.document, 3)
        summary_list = [str(sentence) for sentence in summary]
        result = ' '.join(summary_list)
        return result
    def analyze_text(text):
        return nlp(text)
    @st.cache
    def get_text(raw_url):
        page = urlopen(raw_url)
        soup = BeautifulSoup(page)
        fetched_text = ''.join(map(lambda p: p.text, soup.find_all('p')))
        return fetched_text
    st.subheader = ("Analyze text from url")
    raw_url = st.text_input("enter URL", "Type here")
    text_length = st.slider("Length to preview,50,100")
    if st.button("Extract"):

        if raw_url != "Type here":
            result = get_text(raw_url)
            len_of_full_text = len(result)
            len_of_short_text = round(len(result) / text_length)
            len_of_short_text = round(len(result) / text_length)
            st.info("length::Full Text:: {}".format(len_of_full_text))
            st.info("length::Short Text:: {}".format(len_of_short_text))
            st.write(result[:len_of_short_text])
            summary_docx = sumy_summarizer(result)
            docx = analyze_text(result[:len_of_short_text])
            html = displacy.render(docx, style='ent')
            html = html.replace("\n\n", "\n")
            st.markdown(html, unsafe_allow_html=True)



def main():
    st.title('spacy-streamlit')
    menu = ['Home', 'NER', 'NER without entity', 'URL', 'Text Summary']
    choice = st.sidebar.selectbox("Menu", menu)
    if choice == "NER":
        NER()
    elif choice == "NER without entity":
        Non_NER()
    elif choice == "URL":
        URL()
    else:
        Text_summary()


if __name__ == "__main__":
    main()







