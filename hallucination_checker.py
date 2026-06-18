import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class HallucinationChecker:

    def __init__(self):

        print("Loading embedding model...")

        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    # ----------------------------------------
    # Extract claims
    # ----------------------------------------

    def extract_claims(self, answer):

        sentences = re.split(r"[.?!]", answer)
        return [s.strip() for s in sentences if s.strip()]

    # ----------------------------------------
    # Grounding check
    # ----------------------------------------

    def check_grounding(self, answer, context_chunks):

        if not context_chunks:
            return 0.0, False

        answer_embedding = self.model.encode(answer, convert_to_numpy=True)
        context_embeddings = self.model.encode(context_chunks, convert_to_numpy=True)

        similarities = cosine_similarity(
            [answer_embedding],
            context_embeddings
        )[0]

        max_sim = float(np.max(similarities))
        grounded = max_sim > 0.55

        return max_sim, grounded

    # ----------------------------------------
    # Unsupported claims
    # ----------------------------------------

    def detect_unsupported_claims(self, answer, context_chunks):

        claims = self.extract_claims(answer)
        context_text = " ".join(context_chunks).lower()

        unsupported = []

        for c in claims:
            if c.lower() not in context_text:
                unsupported.append(c)

        return unsupported

    # ----------------------------------------
    # Confidence
    # ----------------------------------------

    def compute_confidence(self, grounding_score, unsupported_count):

        base = grounding_score * 100
        penalty = unsupported_count * 10

        return round(max(0, base - penalty), 2)

    # ----------------------------------------
    # Main validation
    # ----------------------------------------

    def validate(self, answer, context_chunks):

        grounding_score, grounded = self.check_grounding(answer, context_chunks)

        unsupported = self.detect_unsupported_claims(answer, context_chunks)

        confidence = self.compute_confidence(grounding_score, len(unsupported))

        return {
            "confidence": confidence,
            "grounded": grounded,
            "sources_found": len(context_chunks),
            "unsupported_claims": unsupported,
            "grounding_score": round(grounding_score, 3)
        }

    # ----------------------------------------
    # AUTO REPORT GENERATION (.md FILE)
    # ----------------------------------------

    def generate_report(self, answer, context_chunks, filename="hallucination_validation_report.md"):

        result = self.validate(answer, context_chunks)

        report = []

        report.append("# 🧠 Hallucination Validation Report\n")

        report.append("## 1. Answer\n")
        report.append(answer + "\n")

        report.append("## 2. Context\n")
        for i, c in enumerate(context_chunks, 1):
            report.append(f"{i}. {c}")
        report.append("")

        report.append("## 3. Metrics\n")
        report.append(f"- Confidence: {result['confidence']}")
        report.append(f"- Grounded: {result['grounded']}")
        report.append(f"- Grounding Score: {result['grounding_score']}")
        report.append(f"- Sources Found: {result['sources_found']}")
        report.append(f"- Unsupported Claims: {len(result['unsupported_claims'])}\n")

        report.append("## 4. Unsupported Claims\n")

        if result["unsupported_claims"]:
            for c in result["unsupported_claims"]:
                report.append(f"- ❌ {c}")
        else:
            report.append("None ✅")

        report.append("\n## 5. Final Verdict\n")

        if result["grounded"] and len(result["unsupported_claims"]) == 0:
            report.append("🟢 Fully Grounded (No Hallucination Detected)")
        elif result["confidence"] > 70:
            report.append("🟡 Mostly Grounded (Minor Hallucination Risk)")
        else:
            report.append("🔴 High Hallucination Risk")

        # WRITE FILE
        with open(filename, "w", encoding="utf-8") as f:
            f.write("\n".join(report))

        print(f"\n✅ Report generated: {filename}")


# ----------------------------------------
# RUN EXAMPLE
# ----------------------------------------

if __name__ == "__main__":

    checker = HallucinationChecker()

    context = [
        "Employees are entitled to 20 annual leaves per year.",
        "Leave must be approved by the manager.",
        "Vacation policy allows paid time off.",
        "Employees can take sick leave when required."
    ]

    answer = input("\nEnter Model Answer: ")

    checker.generate_report(answer, context)