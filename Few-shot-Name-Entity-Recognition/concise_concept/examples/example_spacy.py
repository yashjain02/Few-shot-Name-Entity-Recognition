


#Resume Extraction:
#pip install PyMuPDF

#Function for converting PDF into plain text:

import re
import spacy
import sys, fitz
def convertToText(fname):

    doc = fitz.open(fname)
    text = ""
    for page in doc:
        text = text + str(page.get_text())
    tx = " ".join(text.split("\n"))
    return tx

tx=convertToText("english.pdf")
tx
#Extracting name:
#For extracting names, pretrained model from spaCy can be downloaded using: --> python -m spacy download en_core_web_lg

#Extracting email and mobile number and Skills:
#Email and mobile numbers have fixed patterns. To extract them regular expression(RegEx) can be used.

import re
import spacy

#loading pretrained model
nlp = spacy.load("en_core_web_lg")
#jsonl file
skills = "skill_patterns.jsonl"

#adding pipe to pretrained model
ruler = nlp.add_pipe("entity_ruler",before="ner")
#for skills extraction
ruler.from_disk(skills)
#for email and mobile extraction
patterns = \
[
    {
        "label": "EMAIL", "pattern": [{"TEXT": {"REGEX": "([^@\s]+@[^@]+\.[^@|\s]+)"}}]
    },
    {
        "label": "Mobile", "pattern": [{"TEXT": {"REGEX": "\d{3}[-\.\s]??\d{3}[-\.\s]??\d{4}|\(\d{3}\)\s*\d{3}[-\.\s]??\d{4}|\d{3}[-\.\s]??\d{4}"}}]
    }
]
ruler.add_patterns(patterns)

#To display the required entities, doc.ents function can be used, each entity has its own label(ent.label_) and text(ent.text).
nlp_model = spacy.load("en_core_web_lg")
doc1 = nlp_model(tx)
dict = {}
skills = []
#extract entities
i=0
ents = list(doc1.ents)
print(ents)
for ent in ents:
    if ent.label_ == 'PERSON' and i==0:
        dict['PERSON'] = ent.text
        i = i+1
    if ent.label_ == 'EMAIL':
        dict['EMAIL'] = ent.text
    if ent.label_ == 'MOBILE':
        dict['MOBILE'] = ent.text
    if ent.label_ == 'SKILL':
        skills.append(ent.text)

skills = [i.capitalize() for i in set([i.lower() for i in skills])]
dict["SKILLS"] = skills