"""Retrieval eval. Compares baseline chunks and moments against caption timestamps. No LLM."""

from __future__ import annotations

import json

from app.config import DATA_DIR
from app.eval_labels import LABELS
from app.search import expand_query, get_index
from app.textutil import format_timestamp


def overlaps(hit: dict, start: float, end: float) -> bool:
    return hit["end"] > start and hit["start"] < end


def covers_gold(hit: dict, gold: list[tuple[float, float]]) -> bool:
    return any(overlaps(hit, start, end) for start, end in gold)


def first_hit_rank(hits: list[dict], gold: list[tuple[float, float]]) -> int | None:
    for rank, hit in enumerate(hits, start=1):
        if covers_gold(hit, gold):
            return rank
    return None


def reciprocal_rank(hits: list[dict], gold: list[tuple[float, float]]) -> float:
    rank = first_hit_rank(hits, gold)
    if rank is None:
        return 0.0
    return 1.0 / rank


def summarize(rows: list[dict], system: str) -> dict:
    total = len(rows) or 1
    hit1 = sum(1 for row in rows if row[system]["hit_at_1"])
    hit4 = sum(1 for row in rows if row[system]["hit_at_4"])
    mrr = sum(row[system]["mrr"] for row in rows) / total
    return {"hit_at_1": hit1 / total, "hit_at_4": hit4 / total, "mrr": mrr, "n": len(rows)}


def evaluate(index=None) -> dict:
    index = index or get_index()
    rows = []
    for label in LABELS:
        match_query = expand_query(label["query"])
        baseline_hits = index.baseline(match_query)
        moment_hits = index.moment(match_query)
        rows.append(
            {
                "id": label["id"],
                "query": label["query"],
                "evidence": label["evidence"],
                "gold": label["gold"],
                "baseline": _score(baseline_hits, label["gold"]),
                "moment": _score(moment_hits, label["gold"]),
            }
        )
    return {
        "rows": rows,
        "baseline": summarize(rows, "baseline"),
        "moment": summarize(rows, "moment"),
    }


def _score(hits: list[dict], gold: list[tuple[float, float]]) -> dict:
    rank = first_hit_rank(hits, gold)
    top = hits[0] if hits else None
    return {
        "hit_at_1": rank == 1,
        "hit_at_4": rank is not None and rank <= 4,
        "mrr": 0.0 if rank is None else 1.0 / rank,
        "rank": rank,
        "top": None
        if top is None
        else {"start": top["start"], "end": top["end"], "title": top.get("title")},
    }


def render_markdown(report: dict) -> str:
    lines = [
        "# Retrieval eval",
        "",
        "Gold windows are caption timestamps. A hit counts when the retrieved span overlaps that window. No language model is called.",
        "",
        "| Query | Baseline top-1 | Moment top-1 | Baseline | Moment |",
        "|---|---|---|---|---|",
    ]
    for row in report["rows"]:
        lines.append(
            "| {query} | {bspan} | {mspan} | {bmark} | {mmark} |".format(
                query=row["query"],
                bspan=_span(row["baseline"]["top"]),
                mspan=_span(row["moment"]["top"]),
                bmark=_mark(row["baseline"]),
                mmark=_mark(row["moment"]),
            )
        )
    lines.extend(["", "| System | Hit@1 | Hit@4 | MRR |", "|---|---:|---:|---:|"])
    for name in ("baseline", "moment"):
        stats = report[name]
        lines.append(
            f"| {name} | {stats['hit_at_1']:.2f} | {stats['hit_at_4']:.2f} | {stats['mrr']:.2f} |"
        )
    lines.extend(["", "Evidence for each gold window:", ""])
    for row in report["rows"]:
        lines.append(f"- {row['id']}: {row['evidence']}")
    lines.append("")
    return "\n".join(lines)


def _span(top: dict | None) -> str:
    if not top:
        return "—"
    return f"{format_timestamp(top['start'])}–{format_timestamp(top['end'])}"


def _mark(scored: dict) -> str:
    if scored["hit_at_1"]:
        return "hit@1"
    if scored["hit_at_4"]:
        return f"hit@{scored['rank']}"
    return "miss"


def main() -> None:
    report = evaluate()
    text = render_markdown(report)
    path = DATA_DIR / "eval_report.md"
    path.write_text(text, encoding="utf-8")
    (DATA_DIR / "eval_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(text)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
