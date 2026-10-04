import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
INDEX_DIR = DATA_DIR / "index"
TRANSCRIPT_PATH = DATA_DIR / "transcript.json"

VIDEO_ID = "zjkBMFhNj_g"
VIDEO_URL = f"https://www.youtube.com/watch?v={VIDEO_ID}"
VIDEO_TITLE = "[1hr Talk] Intro to Large Language Models"
VIDEO_AUTHOR = "Andrej Karpathy"
VIDEO_WHY = (
    "A one-hour technical talk with clear topic shifts: what an LLM is, "
    "pretraining, RLHF, security attacks, and the LLM operating system. "
    "Those shifts are natural moments, so fixed-size chunks and moment "
    "boundaries disagree in a way you can see."
)

SAMPLE_QUERIES = [
    "What is a large language model?",
    "How does RLHF work after pretraining?",
    "What are jailbreaks and prompt injection?",
    "How is an LLM like an operating system?",
]

EXPANSIONS = {
    "rlhf": "reinforcement learning from human feedback",
    "llm": "large language model",
    "llms": "large language models",
}

WORDS_PER_CHUNK = 110
CHUNK_OVERLAP_WORDS = 25
BEAT_SECONDS = 28
MIN_MOMENT_SECONDS = 50
MAX_MOMENT_SECONDS = 180
TOP_K = 4

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
