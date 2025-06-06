import chromadb
from chromadb.config import Settings

class VectorStore:
    def __init__(self, persist_directory="data/vector_db"):
        self.client = chromadb.Client(Settings(persist_directory=persist_directory))
        self.collection = self.client.get_or_create_collection(name="agent_memory")

    def add_memory(self, task_id: str, content: str):
        self.collection.add(documents=[content], ids=[task_id])

    def search(self, query: str):
        return self.collection.query(query_texts=[query], n_results=3)
