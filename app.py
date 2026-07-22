from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List
import re

app = FastAPI(title="Grounded QA API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Chunk(BaseModel):
    chunk_id: str
    text: str

class QARequest(BaseModel):
    question: str
    chunks: List[Chunk] = []

STOPWORDS = {
    "a","an","the","is","are","was","were","in","on","at","of","for",
    "to","and","or","what","when","where","who","which","how","did",
    "does","do","this","that","it","by","released","year"
}

def tokenize(text: str):
    return set(w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOPWORDS)

def best_chunk_match(question: str, chunks: List[Chunk]):
    q_tokens = tokenize(question)
    if not q_tokens:
        return None, 0.0

    scored = []
    for chunk in chunks:
        c_tokens = tokenize(chunk.text)
        if not c_tokens:
            continue
        overlap = q_tokens & c_tokens
        score = len(overlap) / len(q_tokens)
        scored.append((score, chunk))

    if not scored:
        return None, 0.0

    scored.sort(key=lambda x: x[0], reverse=True)
    top_score, top_chunk = scored[0]
    return top_chunk, top_score

def extract_answer(question: str, chunk_text: str):
    sentences = re.split(r'(?<=[.!?])\s+', chunk_text.strip())
    q_tokens = tokenize(question)

    best_sentence = None
    best_overlap = -1
    for sentence in sentences:
        s_tokens = tokenize(sentence)
        overlap = len(q_tokens & s_tokens)
        if overlap > best_overlap:
            best_overlap = overlap
            best_sentence = sentence

    return best_sentence.strip() if best_sentence else chunk_text.strip()

@app.post("/grounded-answer")
async def grounded_answer(request: Request):
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(status_code=200, content={
            "answer": "I don't know",
            "citations": [],
            "confidence": 0.0,
            "answerable": False
        })

    try:
        qa = QARequest(**body)
    except Exception:
        return JSONResponse(status_code=200, content={
            "answer": "I don't know",
            "citations": [],
            "confidence": 0.0,
            "answerable": False
        })

    question = (qa.question or "").strip()
    chunks = qa.chunks or []

    if question == "" or len(chunks) == 0:
        return {
            "answer": "I don't know",
            "citations": [],
            "confidence": 0.0,
            "answerable": False
        }

    valid_ids = {c.chunk_id for c in chunks}

    top_chunk, score = best_chunk_match(question, chunks)

    THRESHOLD = 0.34

    if top_chunk is None or score < THRESHOLD:
        return {
            "answer": "I don't know",
            "citations": [],
            "confidence": round(min(0.3, 0.1 + score * 0.3), 2),
            "answerable": False
        }

    answer_text = extract_answer(question, top_chunk.text)

    if top_chunk.chunk_id not in valid_ids:
        return {
            "answer": "I don't know",
            "citations": [],
            "confidence": 0.2,
            "answerable": False
        }

    confidence = round(min(0.98, 0.5 + score * 0.5), 2)

    return {
        "answer": answer_text,
        "citations": [top_chunk.chunk_id],
        "confidence": confidence,
        "answerable": True
    }

@app.get("/")
def home():
    return {"status": "running", "message": "Grounded QA API"}

@app.get("/grounded-answer")
def grounded_answer_get_guard():
    return JSONResponse(status_code=200, content={
        "message": "This endpoint only accepts POST requests."
    })
