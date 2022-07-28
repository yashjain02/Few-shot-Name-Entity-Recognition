
from pydantic import BaseModel
import spacy
import concise_concepts



app = FastAPI(title='Loan Prediction', version='1.0', description='fastapi for loan pred')


class Data(BaseModel):
    text: str
    fruit: list
    vegetable: list
    meat: list


@app.post('/individualEntry')
def predict_indiviual(data:Data):
    data1=data.dict()
    text=data1['text']
    data1.pop('text')
    doc= predict(text,data1)
    return {'predict': doc}

def predict(text,data1):
    nlp = spacy.load("en_core_web_lg")
    print(data1)
    topn = [50, 50, 150]
    len(topn) == len
    nlp.add_pipe("concise_concepts", config={"data": data1,"topn":[100,100,100]})
    print(text)
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]
if __name__ == '__main__':
    uvicorn.run("FastAPI:app", host="0.0.0.0", port=8000, reload=True)
