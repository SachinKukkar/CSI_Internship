import pandas as pd
import faiss
import os
import openai
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# Load .env file for OpenAI key
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# === STEP 1: Load and prepare dataset ===
df = pd.read_csv("Training Dataset.csv")

# Combine each row into a text document
documents = df.fillna("").astype(str).apply(lambda x: " | ".join(x), axis=1).tolist()

# === STEP 2: Embed documents ===
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(documents, show_progress_bar=True)

# === STEP 3: Create FAISS index ===
dimension = embeddings[0].shape[0]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# === STEP 4: Function to retrieve top K relevant docs ===
def retrieve_relevant_docs(query, k=3):
    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding, k)
    return [documents[i] for i in indices[0]]

# === STEP 5: Generate response using OpenAI ===
def generate_answer(query, context_docs):
    context = "\n".join(context_docs)
    prompt = f"""Answer the following question based on the data provided.
Context:
{context}

Question: {query}
Answer:"""

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful AI assistant that answers questions based on provided context."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=300,
        temperature=0.7,
    )
    return response['choices'][0]['message']['content'].strip()

# === STEP 6: Chat interface ===
def chat():
    print("🔍 Loan Dataset RAG Chatbot")
    print("Type 'exit' to quit.\n")
    while True:
        query = input("You: ")
        if query.lower() in ['exit', 'quit']:
            break
        context = retrieve_relevant_docs(query)
        answer = generate_answer(query, context)
        print(f"🤖: {answer}\n")

if __name__ == "__main__":
    chat()
