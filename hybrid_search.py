import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from sklearn.metrics.pairwise import cosine_similarity


# Sample documents
documents = [
    {
        "document": "employee_policy.pdf",
        "content": "Employees are entitled to 20 annual leaves per year."
    },
    {
        "document": "it_security.pdf",
        "content": "Passwords must be changed every 90 days."
    },
    {
        "document": "benefits_guide.pdf",
        "content": "Health insurance benefits are available to all employees."
    },
    {
        "document": "remote_work_policy.pdf",
        "content": "Employees can work remotely up to three days per week."
    }
]


class HybridSearch:

    def __init__(self, documents):
        self.documents = documents

        # Load embedding model
        print("Loading embedding model...")
        self.model = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )

        # Extract text
        self.texts = [
            doc["content"]
            for doc in documents
        ]

        # Create embeddings
        print("Generating document embeddings...")
        self.doc_embeddings = self.model.encode(
            self.texts,
            convert_to_numpy=True
        )

        # BM25 setup
        tokenized_docs = [
            text.lower().split()
            for text in self.texts
        ]

        self.bm25 = BM25Okapi(tokenized_docs)

    def semantic_search(self, query):
        query_embedding = self.model.encode(
            query,
            convert_to_numpy=True
        )

        similarities = cosine_similarity(
            [query_embedding],
            self.doc_embeddings
        )[0]

        return similarities

    def keyword_search(self, query):
        tokenized_query = query.lower().split()

        return np.array(
            self.bm25.get_scores(tokenized_query)
        )

    def hybrid_search(self, query, top_k=3):

        semantic_scores = self.semantic_search(query)
        keyword_scores = self.keyword_search(query)

        # Normalize semantic scores
        semantic_scores = (
            semantic_scores - semantic_scores.min()
        ) / (
            semantic_scores.max()
            - semantic_scores.min()
            + 1e-8
        )

        # Normalize BM25 scores
        keyword_scores = (
            keyword_scores - keyword_scores.min()
        ) / (
            keyword_scores.max()
            - keyword_scores.min()
            + 1e-8
        )

        # Combine scores
        hybrid_scores = (
            0.7 * semantic_scores
            + 0.3 * keyword_scores
        )

        top_indices = np.argsort(
            hybrid_scores
        )[::-1][:top_k]

        results = []

        for idx in top_indices:
            results.append({
                "document": self.documents[idx]["document"],
                "score": round(
                    float(hybrid_scores[idx]),
                    2
                ),
                "search_type": "hybrid"
            })

        return results


if __name__ == "__main__":

    search_engine = HybridSearch(documents)

    query = input(
        "\nEnter your search query: "
    )

    results = search_engine.hybrid_search(
        query
    )

    print("\nTop Results:\n")

    for result in results:
        print(result)