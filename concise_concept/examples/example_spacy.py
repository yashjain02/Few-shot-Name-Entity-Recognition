

import spacy

import concise_concepts

from data import data, text

nlp = spacy.load("en_core_web_lg")

nlp.add_pipe("concise_concepts", config={"data": data,"topn":[100,100,100,100,100],"ent_score":True})

doc = nlp(text)
print([(ent.text, ent.label_, ent._.ent_score) for ent in doc.ents])