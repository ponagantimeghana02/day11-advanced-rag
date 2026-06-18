from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


# ---------------------------------------------------
# KNOWLEDGE BASE
# ---------------------------------------------------

documents = [
    {
        "text": "Employee notice period is 2 months as per HR policy section 4.2.",
        "metadata": {
            "source": "HR Policy",
            "section": "4.2",
            "department": "HR"
        }
    },
    {
        "text": "Employees are entitled to 20 annual leaves per year.",
        "metadata": {
            "source": "Employee Handbook",
            "section": "3.1",
            "department": "HR"
        }
    },
    {
        "text": "Product installation requires admin privileges.",
        "metadata": {
            "source": "Technical Manual",
            "section": "2.3",
            "department": "IT"
        }
    },
    {
        "text": "Company SOP requires manager approval before travel.",
        "metadata": {
            "source": "Company SOP",
            "section": "5.0",
            "department": "Operations"
        }
    },
    {
        "text": "HR policies are reviewed every year in January.",
        "metadata": {
            "source": "HR Policies",
            "section": "1.0",
            "department": "HR"
        }
    }
]


# ---------------------------------------------------
# ENTERPRISE ASSISTANT
# ---------------------------------------------------

class EnterpriseAssistant:

    def __init__(self):

        print("Loading models...")

        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

        self.history = []

        self.embeddings = self.embedder.encode(
            [d["text"] for d in documents],
            convert_to_numpy=True
        )

    # ---------------------------------------------------
    # SAFE METADATA FILTER
    # ---------------------------------------------------

    def filter_by_metadata(self, docs, department=None):

        if not department:
            return docs

        filtered = []

        for d in docs:

            meta = d.get("metadata", {})

            if meta.get("department", "").lower() == department.lower():
                filtered.append(d)

        return filtered

    # ---------------------------------------------------
    # SEMANTIC SEARCH
    # ---------------------------------------------------

    def semantic_search(self, query):

        query_vec = self.embedder.encode(
            query,
            convert_to_numpy=True
        )

        scores = cosine_similarity(
            [query_vec],
            self.embeddings
        )[0]

        results = []

        for i, score in enumerate(scores):
            results.append({
                "doc": documents[i],
                "score": float(score)
            })

        return results

    # ---------------------------------------------------
    # RERANKING
    # ---------------------------------------------------

    def rerank(self, query, candidates):

        if not candidates:
            return []

        pairs = [
            (query, c["doc"]["text"])
            for c in candidates
        ]

        scores = self.reranker.predict(pairs)

        reranked = sorted(
            zip(candidates, scores),
            key=lambda x: x[1],
            reverse=True
        )

        return reranked

    # ---------------------------------------------------
    # ANSWER GENERATION (SAFE)
    # ---------------------------------------------------

    def generate_answer(self, top_docs):

        if not top_docs:
            return "No relevant information found.", []

        answer_parts = []
        citations = []

        for item in top_docs:

        # SAFE extraction
            doc = item.get("doc", {})

            text = doc.get("text", "")   # SAFE
            meta = doc.get("metadata", {})

            if text:
                answer_parts.append(text)

            citations.append(
                f"{meta.get('source','Unknown')} (Section {meta.get('section','N/A')})"
            )

        answer = "According to internal documents:\n\n" + " ".join(answer_parts)

        return answer, citations

    # ---------------------------------------------------
    # MAIN PIPELINE
    # ---------------------------------------------------

    def ask(self, query, department=None):

        self.history.append(query)

        # Step 1: Semantic Search
        semantic_results = self.semantic_search(query)

        # Step 2: Metadata Filter
        filtered = self.filter_by_metadata(
            semantic_results,
            department
        )

        # 🔥 FIX: fallback if filter removes everything
        if not filtered:
            filtered = semantic_results

        # Step 3: Top 20
        top20 = sorted(
            filtered,
            key=lambda x: x["score"],
            reverse=True
        )[:20]

        if not top20:
            top20 = semantic_results[:20]

        # Step 4: Re-rank
        reranked = self.rerank(query, top20)

        top5 = []

        for item in reranked[:5]:
            candidate = item[0]

            # safety check
            if isinstance(candidate, dict):
                top5.append(candidate)

        # Step 5: Answer
        answer, citations = self.generate_answer(top5)

        return {
            "question": query,
            "answer": answer,
            "citations": citations,
            "history": self.history
        }


# ---------------------------------------------------
# RUN PROGRAM
# ---------------------------------------------------

if __name__ == "__main__":

    assistant = EnterpriseAssistant()

    while True:

        query = input("\nAsk a question (or type exit): ")

        if query.lower() == "exit":
            break

        dept = input("Filter by department (HR/IT/Operations/None): ")

        dept = None if dept.lower() == "none" else dept

        response = assistant.ask(query, dept)

        print("\n" + "=" * 70)
        print("ANSWER")
        print("=" * 70)
        print(response["answer"])

        print("\nCITATIONS:")
        for c in response["citations"]:
            print("-", c)

        print("\nHISTORY:")
        print(response["history"])
        print("=" * 70)