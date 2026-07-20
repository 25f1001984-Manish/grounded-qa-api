from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from google import genai
import os
import json

# -----------------------------
# Gemini Client
# -----------------------------

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

# -----------------------------
# FastAPI App
# -----------------------------

app = FastAPI(title="Grounded QA API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Models
# -----------------------------

class Chunk(BaseModel):
    chunk_id: str
    text: str

class QARequest(BaseModel):
    question: str
    chunks: List[Chunk]

# -----------------------------
# Prompt Builder
# -----------------------------

def build_prompt(question, chunks):

    context = ""

    for chunk in chunks:
        context += f"[{chunk.chunk_id}]\n{chunk.text}\n\n"

    prompt = f"""
You are an expert Grounded Question Answering assistant.

STRICT RULES:

1. Use ONLY the provided context.
2. NEVER use outside knowledge.
3. NEVER guess.
4. NEVER infer missing facts.
5. Cite ONLY chunk IDs provided.
6. If answer cannot be fully answered from context, return EXACTLY:

{{
  "answer":"I don't know",
  "citations":[],
  "confidence":0.2,
  "answerable":false
}}

If answer exists return ONLY JSON like:

{{
  "answer":"...",
  "citations":["C1"],
  "confidence":0.95,
  "answerable":true
}}

Context:

{context}

Question:

{question}

Return ONLY JSON.
"""

    return prompt

# -----------------------------
# API
# -----------------------------

@app.post("/grounded-answer")
def grounded_answer(request: QARequest):

    if request.question.strip() == "" or len(request.chunks) == 0:

        return {
            "answer": "I don't know",
            "citations": [],
            "confidence": 0.0,
            "answerable": False
        }

    prompt = build_prompt(request.question, request.chunks)

    try:

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        output = response.text.strip()

        if output.startswith("```"):
            output = output.replace("```json", "")
            output = output.replace("```", "")
            output = output.strip()

        result = json.loads(output)

    except Exception:

        return {
            "answer": "I don't know",
            "citations": [],
            "confidence": 0.0,
            "answerable": False
        }

    # -----------------------------
    # Validation
    # -----------------------------

    valid_ids = {chunk.chunk_id for chunk in request.chunks}

    citations = result.get("citations", [])

    for cid in citations:
        if cid not in valid_ids:
            return {
                "answer": "I don't know",
                "citations": [],
                "confidence": 0.2,
                "answerable": False
            }

    confidence = result.get("confidence", 0.2)

    try:
        confidence = float(confidence)
    except:
        confidence = 0.2

    confidence = max(0.0, min(1.0, confidence))

    answerable = bool(result.get("answerable", False))

    answer = result.get("answer", "I don't know")

    if not answerable:

        return {
            "answer": "I don't know",
            "citations": [],
            "confidence": min(confidence, 0.3),
            "answerable": False
        }

    if len(citations) == 0:

        return {
            "answer": "I don't know",
            "citations": [],
            "confidence": 0.2,
            "answerable": False
        }

    return {
        "answer": answer,
        "citations": citations,
        "confidence": confidence,
        "answerable": True
    }

# -----------------------------
# Home
# -----------------------------

@app.get("/")
def home():
    return {
        "status": "running",
        "message": "Grounded QA API"
    }
