# Profiling prompts

`t04a.sh --speed --profile` sends each `*.txt` here, in name order, as one user message to llama-server with the hook
and `STRATA_MOE_COUNTS` on, up to `PROFILE_TOKENS` (default 256) tokens, temperature 0, no `ignore_eos`, so the
counts are the routing of natural answers. `hot_set.py --counts` then builds the VRAM hot set and the warm lists from
the decode counts of all of them. The DeepSeek-V4.1 dims make that about 40 layers x 6 experts per decode token, so
six answers of up to 256 tokens give up to 368,640 decode activations, against the 95-token oracle's 22,800.

| File | Kind | What it asks for |
|---|---|---|
| `code-1.txt` | code | a C++17 thread-safe bounded queue |
| `code-2.txt` | code | SQL index advice and a rewrite |
| `prose-1.txt` | prose | a four-paragraph short story |
| `prose-2.txt` | prose | an explanation of vaccines for a teenager |
| `chat-1.txt` | chat | a three-day trip plan |
| `chat-2.txt` | chat | advice on missed sprint deadlines |

None of them is the speed bench's prompt (`Write a Python function to reverse a linked list. Only the function.`)
or the oracle's (`tools/dsv41/oracle/prompt.txt`), so the speed runs measure hot sets built on other text, not
in-sample ones. All prompts are in English, as the bench is. Adding a file adds a request; the plan's estimate counts
each at `PROFILE_TOKENS` / 18 tok/s + 10 s (the hook's measured decode rate on an A100).
