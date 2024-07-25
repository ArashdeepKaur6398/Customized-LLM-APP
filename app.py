import gradio as gr
from huggingface_hub import InferenceClient
from typing import List, Tuple
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

client = InferenceClient("HuggingFaceH4/zephyr-7b-beta")

class FrenchLearningApp:
    def __init__(self) -> None:
        self.documents = []
        self.embeddings = None
        self.index = None
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.load_pdf("EasyFrenchStepByStep.pdf")
        self.build_vector_db()

    def load_pdf(self, file_path: str) -> None:
        doc = fitz.open(file_path)
        self.documents = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            self.documents.append({"page": page_num + 1, "content": text})
        print("Easy French Step-by-Step PDF processed successfully!")

    def build_vector_db(self) -> None:
        self.embeddings = self.model.encode([doc["content"] for doc in self.documents])
        self.index = faiss.IndexFlatL2(self.embeddings.shape[1])
        self.index.add(np.array(self.embeddings))
        print("Vector database built successfully!")

    def search_documents(self, query: str, k: int = 3) -> List[str]:
        query_embedding = self.model.encode([query])
        D, I = self.index.search(np.array(query_embedding), k)
        results = [self.documents[i]["content"] for i in I[0]]
        return results if results else ["No relevant information found."]

app = FrenchLearningApp()

def respond(message: str, history: List[Tuple[str, str]]):
    system_message = """You are a helpful French language tutor based on the book 'Easy French Step-by-Step' by Myrna Bell Rochester. You assist users in learning French grammar, vocabulary, and pronunciation. You can explain concepts, provide examples, and offer practice exercises. You can communicate in both English and French, adapting to the user's level. Always encourage and motivate learners, and provide corrections when necessary. If asked, you can engage in simple conversations in French to help users practice."""
    messages = [{"role": "system", "content": system_message}]

    for user, assistant in history:
        messages.append({"role": "user", "content": user})
        messages.append({"role": "assistant", "content": assistant})

    messages.append({"role": "user", "content": message})

    # RAG - Retrieve relevant documents
    retrieved_docs = app.search_documents(message)
    context = "\n".join(retrieved_docs)
    messages.append({"role": "system", "content": "Relevant information: " + context})

    full_response = ""
    for message in client.chat_completion(
        messages,
        max_tokens=500,
        stream=True,
        temperature=0.7,
        top_p=0.9,
    ):
        token = message.choices[0].delta.content
        if token:
            full_response += token

    yield full_response

demo = gr.Blocks()

with demo:
    gr.Markdown(
        """‼️Disclaimer: This chatbot is based on 'Easy French Step-by-Step' by Myrna Bell Rochester. It's for educational purposes only and not a substitute for a human French teacher or immersive language learning.‼️"""
    )
    
    chatbot = gr.ChatInterface(
        respond,
        examples=[
            ["Can you explain the use of 'être' and 'avoir'?"],
            ["What are some common French greetings?"],
            ["How do I form questions in French?"],
            ["How do I conjugate regular -er verbs in French?"],
            ["What's the difference between 'tu' and 'vous'?"],
            ["Can we have a simple conversation in French about hobbies?"],
        ],
        title='French Practicing Assistant 🇫🇷📚'
    )

if __name__ == "__main__":
    demo.launch()