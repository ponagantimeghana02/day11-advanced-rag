from rank_bm25 import BM25Okapi
from nltk.corpus import wordnet
import nltk

# Download WordNet once
nltk.download("wordnet")
nltk.download("omw-1.4")


class QueryExpansionEngine:

    def __init__(self):
        self.llm_expansion_rules = {
            "leave policy": [
                "employee leave policy",
                "vacation policy",
                "annual leave",
                "sick leave",
                "holiday rules"
            ],
            "health insurance": [
                "medical benefits",
                "health coverage",
                "employee insurance"
            ]
        }

    def generate_synonyms(self, query):
        """
        Generate synonyms using WordNet
        """

        expanded_terms = set()

        for word in query.lower().split():

            expanded_terms.add(word)

            for syn in wordnet.synsets(word):

                for lemma in syn.lemmas():

                    synonym = lemma.name().replace("_", " ")

                    if synonym != word:
                        expanded_terms.add(synonym)

        return list(expanded_terms)

    def llm_assisted_expansion(self, query):
        """
        Simulated LLM expansion
        """

        return self.llm_expansion_rules.get(
            query.lower(),
            []
        )

    def expand_query(self, query):

        synonyms = self.generate_synonyms(
            query
        )

        llm_terms = self.llm_assisted_expansion(
            query
        )

        expanded_query = list(
            dict.fromkeys(
                synonyms + llm_terms
            )
        )

        return " ".join(expanded_query)


class SearchEngine:

    def __init__(self, documents):

        self.documents = documents

        tokenized_docs = [
            doc.lower().split()
            for doc in documents
        ]

        self.bm25 = BM25Okapi(
            tokenized_docs
        )

    def retrieve(self, query, top_k=3):

        tokenized_query = query.lower().split()

        scores = self.bm25.get_scores(
            tokenized_query
        )

        ranked_results = sorted(
            zip(self.documents, scores),
            key=lambda x: x[1],
            reverse=True
        )

        return ranked_results[:top_k]


def print_results(title, results):

    print(f"\n{title}")
    print("-" * len(title))

    for rank, (doc, score) in enumerate(
        results,
        start=1
    ):
        print(
            f"{rank}. Score={score:.2f}"
        )
        print(f"   {doc}\n")


if __name__ == "__main__":

    documents = [
        "Employees are entitled to 20 annual leaves every year.",
        "The vacation policy allows paid time off.",
        "Sick leave requires manager approval.",
        "Holiday rules apply to all employees.",
        "Health insurance benefits are available.",
        "Remote work policy allows work from home."
    ]

    query = input(
        "\nEnter Query: "
    )

    search_engine = SearchEngine(
        documents
    )

    expander = QueryExpansionEngine()

    # Original Retrieval
    original_results = (
        search_engine.retrieve(query)
    )

    # Expanded Query
    expanded_query = (
        expander.expand_query(query)
    )

    # Expanded Retrieval
    expanded_results = (
        search_engine.retrieve(
            expanded_query
        )
    )

    print("\n" + "=" * 60)
    print("QUERY EXPANSION REPORT")
    print("=" * 60)

    print("\nOriginal Query:")
    print(query)

    print("\nExpanded Query:")
    print(expanded_query)

    print_results(
        "Original Retrieval",
        original_results
    )

    print_results(
        "Expanded Retrieval",
        expanded_results
    )