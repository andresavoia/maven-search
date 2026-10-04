# Retrieval eval

Gold windows are caption timestamps. A hit counts when the retrieved span overlaps that window. No language model is called.

| Query | Baseline top-1 | Moment top-1 | Baseline | Moment |
|---|---|---|---|---|
| What is a large language model? | 0:21–1:04 | 0:00–1:48 | hit@1 | hit@1 |
| What does he mean by hallucination? | 10:51–11:21 | 42:17–44:30 | hit@1 | hit@2 |
| How does RLHF work after pretraining? | 17:07–17:44 | 20:55–23:38 | miss | hit@1 |
| What are scaling laws? | 27:36–28:13 | 24:59–27:39 | hit@1 | hit@1 |
| How is an LLM like an operating system? | 43:21–43:58 | 42:17–44:30 | hit@1 | hit@1 |
| What are jailbreaks and prompt injection? | 50:00–50:37 | 49:03–51:42 | hit@1 | hit@1 |

| System | Hit@1 | Hit@4 | MRR |
|---|---:|---:|---:|
| baseline | 0.83 | 0.83 | 0.83 |
| moment | 0.83 | 1.00 | 0.92 |

Evidence for each gold window:

- definition: 0:26 caption: 'language model is just two files'
- hallucination: 10:04 caption: 'like hallucinated'; 11:06 'hallucination'
- rlhf: 22:10 caption: 'reinforcement learning from Human'
- scaling: 25:47 caption: 'call scaling laws'
- os: 42:33 caption: 'kernel process'; 43:37 'operating system'
- attacks: 46:16 caption: 'jailbreak attacks'; 53:13 'prompt injection attack'
