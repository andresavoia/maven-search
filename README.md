# Semantic Search vs Moment RAG

Two retrieval pipelines over the same YouTube transcript:

- **Baseline:** overlapping fixed-size word chunks, dense similarity only.
- **Moment RAG:** topic-bounded moments. A moment scores by its best inner beat plus a summary embedding, then that dense score is fused with BM25. On one talk the dense scores sit close together, so the final order also keeps the raw BM25 score. The answer quotes the top moment and cites its time range.

Video: [Intro to Large Language Models](https://www.youtube.com/watch?v=zjkBMFhNj_g) by Andrej Karpathy. The talk changes subject in long stretches (what an LLM is, pretraining, RLHF, security, the LLM OS), so moment boundaries and fixed windows disagree.

Answers are extractive quotes with timestamps, so a citation cannot be invented. Set `OPENAI_API_KEY` if you want a paraphrased answer; the model still only sees retrieved excerpts.

## Deliverable

The submission is [`slides.html`](slides.html). Open that file in a browser. Arrow keys move between the nine slides: the video and why it was chosen, what Moment RAG is, the baseline, the moment pipeline, the comparison, findings, and the retrieval eval. No API key and no Fly.io deploy.

The search UI is a second HTML file, [`static/index.html`](static/index.html). `make run` serves it at http://127.0.0.1:8000 with the two result columns side by side.

## Run

```bash
make setup
make index
make run
```

Open http://127.0.0.1:8000. The page is `static/index.html`. The slide deck is `slides.html`, opened directly in the browser.

```bash
make test
make demo
make eval
```

`make eval` scores retrieval only. Each question has a gold time window taken from the caption where the phrase is spoken. A hit counts when the retrieved span overlaps that window. Nothing is sent to an LLM. The report is written to `data/eval_report.md`.

On this index, six questions:

| System | Hit@1 | Hit@4 | MRR |
|---|---:|---:|---:|
| Baseline | 0.83 | 0.83 | 0.83 |
| Moment RAG | 0.83 | 1.00 | 0.92 |

Hit@1 is the same. The difference is which miss it is. Baseline never reaches the RLHF line at 22:10 (top hit stays at 17:07). Moment RAG puts that moment first, and only misses hallucination at rank 1 (the right span is rank 2).

`make index` downloads `all-MiniLM-L6-v2` on first use and caches the transcript in `data/transcript.json`.

## What changes in the answers

Same four questions on this talk:

- **What is a large language model?** Both land in the opening minute. The fixed chunk is the tighter window. The moment keeps the setup around the “two files” definition.
- **How does RLHF work after pretraining?** Moment RAG retrieves 20:55–23:38, where he explains comparison labels and reinforcement learning from human feedback. The fixed chunk stops at 17:07, still in finetuning, before that stage.
- **What are jailbreaks and prompt injection?** Both find the suffix attack near 50:00. The moment keeps the wider attack explanation instead of one word window.
- **How is an LLM like an operating system?** The moment starts at the line where he says a model is not a chatbot but the kernel of an operating system (about 42:17). The chunk starts later, inside the diagram.

Moment boundaries still slip. A cut can fall a sentence after the thesis, so each moment keeps one beat of overlap with the previous one. Captions are automatic, so titles are spoken fragments. Without `OPENAI_API_KEY` the answer is a grounded quote, not a paraphrase.

The same index also holds a short paper and a slide deck. One question returns a video timestamp, a paper page, and a deck slide. `POST /api/documents` only enqueues the file and returns `202`; parsing runs on a background worker. Set `ADMIN_TOKEN` in `.env` before calling it.

