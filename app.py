from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import os
import json
from google import genai

# ----------------------------
# OpenAI Client
# ----------------------------

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

# ----------------------------
# FastAPI App
# ----------------------------

app = FastAPI(title="Grounded QA API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------
# Request Models
# ----------------------------

class Chunk(BaseModel):
    chunk_id: str
    text: str

class QARequest(BaseModel):
    question: str
    chunks: List[Chunk]

# ----------------------------
# Prompt Builder
# ----------------------------

def build_prompt(question, chunks):

    context = ""

    for chunk in chunks:
        context += f"[{chunk.chunk_id}]\n{chunk.text}\n\n"

    prompt = f"""
You are a Grounded Question Answering system.

STRICT RULES:

1. Answer ONLY using the context.
2. Never use outside knowledge.
3. If the answer is not fully supported by the context, return:

{{
"answer":"I don't know",
"citations":[],
"confidence":0.2,
"answerable":false
}}

4. If answer exists, return ONLY JSON:

{{
"answer":"...",
"citations":["C1"],
"confidence":0.95,
"answerable":true
}}

5. Citation IDs MUST exactly match IDs from the context.

6. Never invent citations.

Context:

{context}

Question:

{question}
"""

    return prompt

# ----------------------------
# Endpoint
# ----------------------------

@app.post("/grounded-answer")
def grounded_answer(request: QARequest):

    # Empty input handling

    if request.question.strip() == "" or len(request.chunks) == 0:

        return {
            "answer": "I don't know",
            "citations": [],
            "confidence": 0.0,
            "answerable": False
        }

    prompt = build_prompt(request.question, request.chunks)

    try:

        response = client.chat.completions.create(

            model="gpt-4.1-mini",

            messages=[
                {
                    "role": "system",
                    "content": "Return JSON only."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],

            temperature=0

        )

        output = response.choices[0].message.content

        result = json.loads(output)

    except Exception:

        return {
            "answer": "I don't know",
            "citations": [],
            "confidence": 0.0,
            "answerable": False
        }

    # ----------------------------
    # Validation
    # ----------------------------

    valid_ids = {chunk.chunk_id for chunk in request.chunks}

    citations = result.get("citations", [])

    # Invalid citation IDs

    for cid in citations:
        if cid not in valid_ids:
            return {
                "answer": "I don't know",
                "citations": [],
                "confidence": 0.2,
                "answerable": False
            }

    confidence = float(result.get("confidence", 0))

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

    return {
        "answer": answer,
        "citations": citations,
        "confidence": confidence,
        "answerable": True
    }

# ----------------------------
# Health Check
# ----------------------------

@app.get("/")
def home():
    return {
        "message": "Grounded QA API is running."
    }
