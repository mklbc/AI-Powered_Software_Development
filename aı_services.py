from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import pipeline
import uvicorn

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"], 
)


@app.get("/")
def home():
    return {"message": "FastAPI service is running! Use /analyze/ endpoint for analysis."}


classifier = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")


class TextInput(BaseModel):
    text: str

# Analiz endpoint'i
@app.post("/analyze/")
def analyze_text(input: TextInput):
    result = classifier(input.text)
    return {"label": result[0]['label'], "score": result[0]['score']}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
