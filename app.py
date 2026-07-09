import logging
from dotenv import load_dotenv
import os
from google import genai
import chromadb
from fastapi import FastAPI
from pydantic import BaseModel

from config import GEMINI_LLM_MODEL, GEMINI_EMBEDDING_MODEL, CHUNK_SIZE, DEFAULT_N_RESULTS

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="documents")

app = FastAPI()


class Document:
    def __init__(self, filepath):
        self.filepath = filepath
        try:
            with open(self.filepath, "r") as file:
                self.content = file.read()
        except FileNotFoundError:
            self.content = None
            logger.warning(f"File not found: {self.filepath}")

    def word_count(self):
        if self.content is None:
            return 0
        return len(self.content.split())

    def get_embedding(self):
        if self.content is None:
            return None
        result = client.models.embed_content(
            model=GEMINI_EMBEDDING_MODEL,
            contents=self.content
        )
        return result.embeddings[0].values


def chunk_text(text, chunk_size=CHUNK_SIZE):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)
    return chunks


def ask_llm(question):
    try:
        response = client.models.generate_content(
            model=GEMINI_LLM_MODEL,
            contents=question
        )
        return response.text
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        return None


def ask_with_context(context, question):
    prompt = f"""
You are a helpful assistant that answers ONLY using the information given below.
If the answer isn't in the text, say "I don't know based on the given information."

Text: {context}

Question: {question}
"""
    return ask_llm(prompt)


class Question(BaseModel):
    question: str


class UploadRequest(BaseModel):
    filepath: str


@app.get("/")
def read_root():
    return {"message": "DocMind API is running"}


@app.post("/ask")
def ask(q: Question):
    result = client.models.embed_content(model=GEMINI_EMBEDDING_MODEL, contents=q.question)
    question_embedding = result.embeddings[0].values

    results = collection.query(query_embeddings=[question_embedding], n_results=DEFAULT_N_RESULTS)
    retrieved_chunks = results["documents"][0]
    combined_context = " ".join(retrieved_chunks)

    answer = ask_with_context(combined_context, q.question)
    return {"answer": answer}


@app.post("/upload")
def upload_document(req: UploadRequest):
    doc = Document(req.filepath)
    if doc.content is None:
        return {"error": "File not found"}

    chunks = chunk_text(doc.content)

    for i, chunk in enumerate(chunks):
        result = client.models.embed_content(model=GEMINI_EMBEDDING_MODEL, contents=chunk)
        embedding = result.embeddings[0].values
        collection.add(
            documents=[chunk],
            embeddings=[embedding],
            ids=[f"{req.filepath}_chunk_{i}"]
        )

    return {"message": f"Uploaded and indexed {len(chunks)} chunks from {req.filepath}"}


@app.get("/documents")
def get_document_count():
    count = collection.count()
    return {"total_chunks_stored": count}