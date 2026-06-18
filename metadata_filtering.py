class MetadataFilterEngine:

    def __init__(self):

        # Documents with metadata
        self.documents = [
            {
                "text": "Employees are entitled to 20 annual leaves per year.",
                "metadata": {
                    "department": "HR",
                    "country": "India",
                    "year": "2025"
                }
            },
            {
                "text": "Financial audit rules are updated yearly.",
                "metadata": {
                    "department": "Finance",
                    "country": "India",
                    "year": "2025"
                }
            },
            {
                "text": "Remote work policy allows 3 days WFH.",
                "metadata": {
                    "department": "HR",
                    "country": "USA",
                    "year": "2024"
                }
            },
            {
                "text": "Tax regulations for enterprises are strict.",
                "metadata": {
                    "department": "Finance",
                    "country": "USA",
                    "year": "2025"
                }
            }
        ]

    # -------------------------------
    # Filter Logic
    # -------------------------------

    def filter_docs(self, department=None, country=None, year=None):

        results = []

        for doc in self.documents:

            meta = doc["metadata"]

            if department and meta["department"].lower() != department.lower():
                continue

            if country and meta["country"].lower() != country.lower():
                continue

            if year and meta["year"] != str(year):
                continue

            results.append(doc)

        return results

    # -------------------------------
    # Query Parser (simple rule-based)
    # -------------------------------

    def parse_query(self, query):

        query = query.lower()

        filters = {
            "department": None,
            "country": None,
            "year": None
        }

        # Department detection
        if "hr" in query:
            filters["department"] = "HR"
        elif "finance" in query:
            filters["department"] = "Finance"

        # Country detection
        if "india" in query:
            filters["country"] = "India"
        elif "usa" in query:
            filters["country"] = "USA"

        # Year detection (simple)
        for y in ["2024", "2025", "2023"]:
            if y in query:
                filters["year"] = y

        return filters

    # -------------------------------
    # Search Function
    # -------------------------------

    def search(self, query):

        filters = self.parse_query(query)

        return self.filter_docs(
            department=filters["department"],
            country=filters["country"],
            year=filters["year"]
        )


# -------------------------------
# Main Execution
# -------------------------------

if __name__ == "__main__":

    engine = MetadataFilterEngine()

    query = input("\nEnter Query: ")

    results = engine.search(query)

    print("\nFILTERED RESULTS:\n")

    if not results:
        print("No matching documents found.")
    else:
        for i, doc in enumerate(results, 1):
            print(f"{i}. {doc['text']}")
            print(f"   Metadata: {doc['metadata']}\n")
            