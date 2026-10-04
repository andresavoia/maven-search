"""Write a short paper and a slide deck so cross-source search has page and slide locators."""

from pathlib import Path

from fpdf import FPDF

from app.config import ROOT

CORPUS_DIR = ROOT / "data" / "corpus"

PAPER = {
    "id": "paper-llm-notes",
    "kind": "paper",
    "title": "Notes on Language Model Training",
    "filename": "llm-notes.pdf",
    "pages": [
        (
            "A large language model is stored as two files: the parameters and a tokenizer. "
            "The parameters are the learned weights. The tokenizer turns text into tokens the network can read."
        ),
        (
            "Pretraining teaches the model to predict the next token on a huge pile of internet text. "
            "Finetuning then trains on curated question and answer documents so the model follows instructions."
        ),
        (
            "Reinforcement learning from human feedback is stage three, after pretraining and finetuning. "
            "Human labelers compare two candidate answers because comparison is easier than writing an answer. "
            "A reward model is trained on those comparison labels, and the language model is optimized against that reward."
        ),
        (
            "A jailbreak is an adversarial suffix of unreadable text that makes a model ignore its safety training. "
            "Prompt injection hides new instructions inside untrusted data, such as a web page, an image, or a shared document. "
            "Data poisoning plants a trigger in the training set so a later phrase changes the model's behavior."
        ),
        (
            "An LLM operating system treats the model as the kernel process rather than a chatbot. "
            "The kernel coordinates memory, tools, and a user interface the way a desktop operating system coordinates processes."
        ),
    ],
}

DECK = {
    "id": "deck-llm-slides",
    "kind": "deck",
    "title": "LLM Talk Slides",
    "filename": "llm-slides.pdf",
    "pages": [
        "Slide 1. A large language model is two files: weights and a tokenizer.",
        "Slide 2. Pretraining predicts the next token. Finetuning teaches instruction following.",
        "Slide 3. RLHF. Humans compare answers. A reward model scores them. The policy is updated from that reward.",
        "Slide 4. Security. A jailbreak suffix bypasses safety training. Prompt injection hides instructions in untrusted content.",
        "Slide 5. LLM OS. The model is the kernel. It coordinates memory, tools, and the interface.",
    ],
}


def _write_pdf(path: Path, title: str, pages: list[str]) -> None:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=18)
    for index, body in enumerate(pages, start=1):
        pdf.add_page()
        pdf.set_font("Helvetica", size=16)
        pdf.multi_cell(0, 10, f"{title} - {index}")
        pdf.ln(4)
        pdf.set_font("Helvetica", size=12)
        pdf.multi_cell(0, 8, body)
    path.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(path)


def ensure_corpus() -> list[dict]:
    sources = []
    for spec in (PAPER, DECK):
        path = CORPUS_DIR / spec["filename"]
        if not path.exists():
            _write_pdf(path, spec["title"], spec["pages"])
        sources.append({**spec, "path": path})
    return sources
