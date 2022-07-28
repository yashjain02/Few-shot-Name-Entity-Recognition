import streamlit as st
import spacy_streamlit as sst
import concise_concepts
from spacy import displacy
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from bs4 import BeautifulSoup
from urllib.request import urlopen


def NER():
    domain = st.selectbox('domain', ['recipe', 'Banking', 'Finance', 'Medical'])
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
        product_list['product'] = st.multiselect("product", ["Card", 'Debit Card'])
        print(product_list)
        if st.button('predict'):
            nlp = spacy.load('en_core_web_lg', disable=['ner'])
            nlp.add_pipe("concise_concepts",
                         config={"data": product_list, "ent_score": True})
            doc = nlp(raw_text)
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
    doc = nlp(raw_text)
    sst.visualize_ner(doc, show_table=False)


def Text_summary():
    nlp=spacy.load("en_core_web_lg")
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

    st.subheader = "Analyze text from url"
    raw_url = st.text_input("enter URL")
    agree = st.checkbox('Paragraph filter')
    if agree:
        if st.button("Extract"):
            if raw_url != "Type here":
                result = get_text(raw_url)
                doc = nlp(result)
                sst.visualize_ner(doc, show_table=False)
    else:
        text_length = st.slider("Length to preview,50,100")
        if st.button("Extract"):

            if raw_url != "Type here":
                result = get_text(raw_url)
                len_of_full_text = len(result)
                len_of_short_text = round(len(result) / text_length)
                st.info("length::Full Text:: {}".format(len_of_full_text))
                st.info("length::Short Text:: {}".format(len_of_short_text))
                st.write(result[:len_of_short_text])
                summary_docx = sumy_summarizer(result)
                docx = analyze_text(result[:len_of_short_text])
                html = displacy.render(summary_docx, style='ent')
                html = html.replace("\n\n", "\n")
                st.markdown(html, unsafe_allow_html=True)
def CV():

    upload_file = st.file_uploader('Choose your CV',type="pdf")
    text = " "
    def convertToText(fname):
        doc = fitz.open(fname)
        text = ""
        for page in doc:
            text = text + str(page.get_text())
        tx = " ".join(text.split("\n"))
        return tx

    nlp = spacy.load("en_core_web_lg")
    tx = convertToText(upload_file)
    skills = "skill_patterns.jsonl"
    ruler = nlp.add_pipe("entity_ruler",before="ner")
    ruler.from_disk(skills)
    pattern = [{
        "label":"EMAIL","pattern":[{"text":{"REGEX":"([^@\s]+@[^@]+\.[^@|\s]+)"}}]
    },
        {
            "label": "Mobile", "pattern": [{"TEXT": {"REGEX": "\d{3}[-\.\s]??\d{3}[-\.\s]??\d{4}|\(\d{3}\)\s*\d{3}[-\.\s]??\d{4}|\d{3}[-\.\s]??\d{4}"}}]
        }]
    ruler.add_patterns(pattern)
    doc1 = nlp(tx)
    sst.visualize_ner(doc1,show_table=False)


def Home():
    st.title("the vectors")


def main():
    st.title('spacy-streamlit')
    menu = ['Home', 'NER', 'No-Class NER', 'Entity in URL',"CV"]
    choice = st.sidebar.selectbox("Menu", menu)
    if choice == "NER":
        NER()
    elif choice == "NER without entity":
        Non_NER()
    elif choice == "CV":
        CV()
    elif choice == "Text_summary":
        Text_summary()
    else:
        Home()



if __name__ == "__main__":
    main()
