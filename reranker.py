from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


# -------------------------------------------------
# Sample Documents
# -------------------------------------------------

documents = [
    "Employees are entitled to 20 annual leaves per year.",
    "Vacation policy allows paid time off for employees.",
    "Sick leave requires manager approval.",
    "Holiday rules apply to all employees.",
    "Remote work policy allows 3 days work from home.",
    "Health insurance benefits are provided to employees.",
    "Finance audit rules are updated yearly.",
    "Employees must follow IT security guidelines.",
    "Passwords must be changed every 90 days.",
    "Company provides maternity and paternity leave benefits.",
    "Employee performance is reviewed annually.",
    "Work hours are 9 AM to 6 PM.",
    "Overtime policy is strictly regulated.",
    "Employees can request unpaid leave.",
    "Training programs are available for skill development.",
    "HR handles employee grievances.",
    "Finance department manages payroll processing.",
    "Employees must submit leave applications in advance.",
    "Remote work approval is required from manager.",
    "Health and safety rules must be followed."
]


# -------------------------------------------------
# Reranker Pipeline
# -------------------------------------------------

class RerankerPipeline:

    def __init__(self):

        print("Loading models...")

        self.embedder = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )

        # Cross Encoder (re-ranker)
        self.cross_encoder = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )

        self.doc_embeddings = self.embedder.encode(
            documents,
            convert_to_numpy=True
        )

    # -------------------------
    # Stage 1: Fast Retrieval
    # -------------------------

    def retrieve_top20(self, query):

        query_vec = self.embedder.encode(
            query,
            convert_to_numpy=True
        )

        scores = cosine_similarity(
            [query_vec],
            self.doc_embeddings
        )[0]

        top_idx = np.argsort(scores)[::-1][:20]

        return [
            (documents[i], float(scores[i]))
            for i in top_idx
        ]

    # -------------------------
    # Stage 2: Cross Encoder Re-ranking
    # -------------------------

    def rerank(self, query, candidates):

        pairs = [
            (query, doc)
            for doc, _ in candidates
        ]

        rerank_scores = self.cross_encoder.predict(pairs)

        reranked = sorted(
            zip(candidates, rerank_scores),
            key=lambda x: x[1],
            reverse=True
        )

        return reranked

    # -------------------------
    # Stage 3: Final Top-K
    # -------------------------

    def search(self, query, top_k=5):

        # Stage 1
        retrieved = self.retrieve_top20(query)

        # Stage 2
        reranked = self.rerank(query, retrieved)

        # Stage 3
        final_results = []

        for (doc, old_score), new_score in reranked[:top_k]:

            final_results.append({
                "document": doc,
                "retrieval_score": round(old_score, 3),
                "rerank_score": float(round(new_score, 3)),
                "pipeline": "reranked"
            })

        return final_results


# -------------------------------------------------
# Evaluation Report
# -------------------------------------------------

def evaluation_report(query, top20, top5):

    print("\n" + "=" * 70)
    print("RERANKING PIPELINE EVALUATION REPORT")
    print("=" * 70)

    print("\nQUERY:")
    print(query)

    print("\n------------------------------")
    print("STAGE 1: TOP 20 RETRIEVAL")
    print("------------------------------")

    for i, (doc, score) in enumerate(top20[:10], 1):
        print(f"{i}. Score={score:.3f} | {doc}")

    print("\n------------------------------")
    print("STAGE 3: FINAL TOP 5 AFTER RERANKING")
    print("------------------------------")

    for i, item in enumerate(top5, 1):
        print(
            f"{i}. RerankScore={item['rerank_score']:.3f} | {item['document']}"
        )

    print("\nInsight:")
    print(
        "Re-ranking improves semantic relevance by using cross-encoder "
        "to evaluate query-document pairs more precisely."
    )

    print("=" * 70)


# -------------------------------------------------
# Main
# -------------------------------------------------

if __name__ == "__main__":

    pipeline = RerankerPipeline()

    query = input("\nEnter Query: ")

    # Stage 1 output
    top20 = pipeline.retrieve_top20(query)

    # Stage 2 + 3 output
    top5 = pipeline.search(query)

    # Evaluation report
    evaluation_report(query, top20, top5)