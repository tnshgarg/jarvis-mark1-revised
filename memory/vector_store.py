# memory/vector_store.py
import chromadb
from memory.base import BaseMemory

class VectorStore(BaseMemory):
    def __init__(self):
        self.client = chromadb.Client()
        self.collection = self.client.get_or_create_collection("long_term")

    def add_memory(self, task_id: str, content: str):
        self.collection.add(documents=[content], ids=[task_id])

    def search_memory(self, query: str, top_k: int = 3):
        results = self.collection.query(query_texts=[query], n_results=top_k)
        return results['documents'][0] if results else []

    def clear_memory(self):
        self.collection.delete()
