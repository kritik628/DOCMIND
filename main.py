from dotenv import load_dotenv
import os
from google import genai

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

print("Client ready")

class Document:
    def __init__(self,filepath):
        self.filepath=filepath
        try:
            with open(self.filepath,"r") as file:
                self.content=file.read()
        except FileNotFoundError:
            self.content=None
            print("FILE NOT FOUND")

    def word_count(self):
        if self.content is None:
            return 0
        return len(self.content.split())

    def get_embedding(self):
        if self.content is None:
            return None
        result = client.models.embed_content(
                model="gemini-embedding-001",
                contents=self.content



        )    
        return result.embeddings[0].values

doc = Document("notes.txt")
embedding = doc.get_embedding()
print(len(embedding))     

import chromadb
chroma_client = chromadb.PersistentClient(path="./chroma_db")
##persistantclient..... helps in storing embdding permanantly in desired location
collection = chroma_client.get_or_create_collection(name="documents")

collection.add(
    documents=[doc.content],
    embeddings=[embedding],
    ids=["doc1"]
)
##documents=[...] → the actual text (ChromaDB stores this alongside the vector, so you can retrieve the original text later, not just numbers)
##=[...] → the vector you already computed
##ids=["doc1"] → a unique identifier for this entry — required, must be unique per item
doc2 = Document("notes2.txt")
embedding2 = doc2.get_embedding()
results = collection.query(
    query_embeddings=[embedding2],
    n_results=1
)
print(results)
results_self = collection.query(
    query_embeddings=[embedding],
    n_results=1
)
print(results_self)

def chunk_text(text,chunk_size=200):
    words=text.split()
    chunks=[]
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)
    return chunks

doc3 = Document("longtext.txt")
chunks = chunk_text(doc3.content, chunk_size=50)
print(len(chunks))
for c in chunks:
    print(len(c.split()))

for i, chunk in enumerate(chunks):
    result = client.models.embed_content(model="gemini-embedding-001", contents=chunk)
    embedding = result.embeddings[0].values
    collection.add(
        documents=[chunk],
        embeddings=[embedding],
        ids=[f"chunk_{i}"]
    )
    print(f"Added chunk {i}")

question = client.models.embed_content(model="gemini-embedding-001", contents="What kind of substance is Alcohol?")
embeddings1=question.embeddings[0].values
results4 = collection.query(
    query_embeddings=[embeddings1],
    n_results=2
)
print(results4)

def ask_llm(question):
  try:
    response=client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=question
    )
    return response.text
  except Exception as e:
    print("LLM call failed:",e)
    return None
    
def ask_with_context(context, question):
    prompt = f"""
You are a helpful assistant that answers ONLY using the information given below.
If the answer isn't in the text, say "I don't know based on the given information."

Text: {context}

Question: {question}
"""
    return ask_llm(prompt) 

retrieved_chunks = results4["documents"][0]
combined_context = " ".join(retrieved_chunks)
answer = ask_with_context(combined_context, "What kind of substance is Alcohol?")
print(answer)
 
