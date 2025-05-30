from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models and data
try:
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    llm = genai.GenerativeModel("gemini-2.0-flash")
    index = faiss.read_index("context.index")
    contexts = np.load("contexts.npy", allow_pickle=True)
    embed_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
except Exception as e:
    print(f"Error loading models: {e}")
    raise

class Query(BaseModel):
    question: str
    top_k: int = 1

def generate_answer(query: str, top_k=1):
    try:
        query_vec = embed_model.encode([query])
        D, I = index.search(query_vec, top_k)
        retrieved_context = "\n".join([contexts[i] for i in I[0]])

        prompt = f"""
Câu hỏi: {query}
Nội dung nội quy liên quan:
{retrieved_context}

Trả lời ngắn gọn, chính xác dựa trên nội dung trên:
"""
        response = llm.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask")
async def ask_question(query: Query):
    try:
        answer = generate_answer(query.question, query.top_k)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Railway will use the PORT environment variable
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)