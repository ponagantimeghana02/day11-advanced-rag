from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class BlackRothAssistant:

    def __init__(self):

        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

        self.documents = []
        self.metadata = []

        self.chat_history = []

    # -----------------------------
    # UPLOAD DOCUMENTS
    # -----------------------------
    def upload_documents(self, docs):
        for d in docs:
            self.documents.append(d["text"])
            self.metadata.append(d["metadata"])

    # -----------------------------
    # AUTO CHUNKING (simple)
    # -----------------------------
    def chunk_text(self, text, chunk_size=100):
        return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

    # -----------------------------
    # HYBRID SEARCH
    # -----------------------------
    def hybrid_search(self, query):

        # BM25
        tokenized_docs = [doc.split() for doc in self.documents]
        bm25 = BM25Okapi(tokenized_docs)
        bm25_scores = bm25.get_scores(query.split())

        # Vector search
        query_vec = self.embedder.encode(query, convert_to_numpy=True)
        doc_vecs = self.embedder.encode(self.documents, convert_to_numpy=True)

        vector_scores = cosine_similarity([query_vec], doc_vecs)[0]

        # Combine scores
        final_scores = 0.5 * np.array(bm25_scores) + 0.5 * vector_scores

        top_idx = np.argsort(final_scores)[::-1][:5]

        results = [
            {
                "text": self.documents[i],
                "score": float(final_scores[i]),
                "metadata": self.metadata[i]
            }
            for i in top_idx
        ]

        return results

    # -----------------------------
    # RE-RANKING
    # -----------------------------
    def rerank(self, query, docs):

        pairs = [(query, d["text"]) for d in docs]
        scores = self.reranker.predict(pairs)

        reranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)

        return [r[0] for r in reranked]

    # -----------------------------
    # RBAC FILTER
    # -----------------------------
    def rbac_filter(self, docs, role):

        return [
            d for d in docs
            if role in d["metadata"].get("access_roles", ["employee"])
        ]

    # -----------------------------
    # CHAT FUNCTION
    # -----------------------------
    def chat(self, query, role="employee"):

        self.chat_history.append(query)

        retrieved = self.hybrid_search(query)

        filtered = self.rbac_filter(retrieved, role)

        reranked = self.rerank(query, filtered)

        top_docs = reranked[:3]

        context = "\n".join([d["text"] for d in top_docs])

        answer = f"Answer based on internal documents:\n\n{context}"

        citations = [
            d["metadata"].get("source", "Unknown")
            for d in top_docs
        ]

        return {
            "answer": answer,
            "citations": citations,
            "chat_history": self.chat_history
        }


# -----------------------------
# RUN EXAMPLE
# -----------------------------
if __name__ == "__main__":

    assistant = BlackRothAssistant()

    assistant.upload_documents([
        {
            "text": "Employees have 2 months notice period as per HR policy.",
            "metadata": {
                "source": "HR Policy",
                "access_roles": ["employee", "hr"]
            }
        },
        {
            "text": "Payroll is processed on 5th of every month.",
            "metadata": {
                "source": "Payroll SOP",
                "access_roles": ["finance", "hr"]
            }
        },
        {
            "text": "Client contracts require legal approval.",
            "metadata": {
                "source": "Contracts",
                "access_roles": ["legal"]
            }
        }
    ])

    while True:

        q = input("\nAsk: ")
        role = input("Role (employee/hr/finance/legal): ")

        result = assistant.chat(q, role)

        print("\nANSWER:\n", result["answer"])
        print("\nCITATIONS:", result["citations"])
        print("\nHISTORY:", result["chat_history"]) 