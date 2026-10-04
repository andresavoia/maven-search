"""Labeled timestamps taken from the caption file, not from retrieval output."""

# Each gold window is the stretch of the talk that contains the answer.
# Times are seconds from the start of the video.
LABELS = [
    {
        "id": "definition",
        "query": "What is a large language model?",
        "gold": [(20, 210)],
        "evidence": "0:26 caption: 'language model is just two files'",
    },
    {
        "id": "hallucination",
        "query": "What does he mean by hallucination?",
        "gold": [(600, 750)],
        "evidence": "10:04 caption: 'like hallucinated'; 11:06 'hallucination'",
    },
    {
        "id": "rlhf",
        "query": "How does RLHF work after pretraining?",
        "gold": [(1260, 1440)],
        "evidence": "22:10 caption: 'reinforcement learning from Human'",
    },
    {
        "id": "scaling",
        "query": "What are scaling laws?",
        "gold": [(1530, 1710)],
        "evidence": "25:47 caption: 'call scaling laws'",
    },
    {
        "id": "os",
        "query": "How is an LLM like an operating system?",
        "gold": [(2540, 2760)],
        "evidence": "42:33 caption: 'kernel process'; 43:37 'operating system'",
    },
    {
        "id": "attacks",
        "query": "What are jailbreaks and prompt injection?",
        "gold": [(2776, 3120), (3193, 3540)],
        "evidence": "46:16 caption: 'jailbreak attacks'; 53:13 'prompt injection attack'",
    },
]
