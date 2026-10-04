import json

from app.config import DATA_DIR, SAMPLE_QUERIES
from app.search import get_index


def main() -> None:
    index = get_index()
    report = []
    for query in SAMPLE_QUERIES:
        result = index.ask(query)
        report.append(
            {
                "query": query,
                "baseline": {
                    "answer": result["baseline"]["answer"],
                    "spans": [
                        {"start": hit["start"], "end": hit["end"], "text": hit["text"][:180]}
                        for hit in result["baseline"]["hits"]
                    ],
                },
                "moment": {
                    "answer": result["moment"]["answer"],
                    "spans": [
                        {
                            "start": hit["start"],
                            "end": hit["end"],
                            "title": hit["title"],
                            "text": hit["text"][:180],
                        }
                        for hit in result["moment"]["hits"]
                    ],
                },
            }
        )
        print("=" * 80)
        print(query)
        print("-- baseline --")
        print(result["baseline"]["answer"])
        print("-- moment --")
        print(result["moment"]["answer"])
    path = DATA_DIR / "comparison.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
