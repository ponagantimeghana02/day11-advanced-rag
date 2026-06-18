from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class RAGEvaluator:

    def __init__(self):

        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    # ---------------------------------------------------
    # 1. Retrieval Precision
    # ---------------------------------------------------
    def retrieval_precision(self, retrieved_docs, relevant_docs):

        if not retrieved_docs:
            return 0.0

        relevant_set = set(relevant_docs)
        retrieved_set = set(retrieved_docs)

        true_positives = len(retrieved_set & relevant_set)

        return true_positives / len(retrieved_docs)

    # ---------------------------------------------------
    # 2. Retrieval Recall
    # ---------------------------------------------------
    def retrieval_recall(self, retrieved_docs, relevant_docs):

        if not relevant_docs:
            return 0.0

        relevant_set = set(relevant_docs)
        retrieved_set = set(retrieved_docs)

        true_positives = len(retrieved_set & relevant_set)

        return true_positives / len(relevant_set)

    # ---------------------------------------------------
    # 3. Context Relevance
    # ---------------------------------------------------
    def context_relevance(self, query, contexts):

        if not contexts:
            return 0.0

        query_emb = self.model.encode(query, convert_to_numpy=True)
        ctx_emb = self.model.encode(contexts, convert_to_numpy=True)

        scores = cosine_similarity([query_emb], ctx_emb)[0]

        return float(np.mean(scores))

    # ---------------------------------------------------
    # 4. Answer Relevance
    # ---------------------------------------------------
    def answer_relevance(self, query, answer):

        query_emb = self.model.encode(query, convert_to_numpy=True)
        ans_emb = self.model.encode(answer, convert_to_numpy=True)

        return float(cosine_similarity([query_emb], [ans_emb])[0][0])

    # ---------------------------------------------------
    # 5. Groundedness
    # ---------------------------------------------------
    def groundedness(self, answer, contexts):

        if not contexts:
            return 0.0

        ans_emb = self.model.encode(answer, convert_to_numpy=True)
        ctx_emb = self.model.encode(contexts, convert_to_numpy=True)

        scores = cosine_similarity([ans_emb], ctx_emb)[0]

        return float(np.max(scores))

    # ---------------------------------------------------
    # 6. Hallucination Rate
    # ---------------------------------------------------
    def hallucination_rate(self, groundedness_score):

        return float(1 - groundedness_score)

    # ---------------------------------------------------
    # FULL EVALUATION PIPELINE
    # ---------------------------------------------------
    def evaluate(self, query, retrieved_docs, relevant_docs, contexts, answer):

        precision = self.retrieval_precision(retrieved_docs, relevant_docs)
        recall = self.retrieval_recall(retrieved_docs, relevant_docs)
        ctx_rel = self.context_relevance(query, contexts)
        ans_rel = self.answer_relevance(query, answer)
        grounded = self.groundedness(answer, contexts)
        hallucination = self.hallucination_rate(grounded)

        return {
            "retrieval_precision": round(precision, 3),
            "retrieval_recall": round(recall, 3),
            "context_relevance": round(ctx_rel, 3),
            "answer_relevance": round(ans_rel, 3),
            "groundedness": round(grounded, 3),
            "hallucination_rate": round(hallucination, 3)
        }


# -------------------------------
# Example Run
# -------------------------------
if __name__ == "__main__":

    evaluator = RAGEvaluator()

    query = "What is notice period?"

    retrieved_docs = [
        "Employee notice period is 2 months",
        "HR policies updated yearly"
    ]

    relevant_docs = [
        "Employee notice period is 2 months"
    ]

    contexts = retrieved_docs

    answer = "The notice period is 2 months as per HR policy."

    result = evaluator.evaluate(
        query,
        retrieved_docs,
        relevant_docs,
        contexts,
        answer
    )
    # metrics = evaluator.evaluate(...)

with open("rag_metrics_report.md", "w") as f:
    f.write("# RAG Metrics Report\n\n")
    f.write(str(result))

    print("\nRAG EVALUATION METRICS:\n")
    for k, v in result.items():
        print(f"{k}: {v}")