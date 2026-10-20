# Provenance of tools/dsv41/quality/

The quality method of the DeepSeek-V4.1-Flash engine is the one an earlier vLLM-EXL3 host-offload experiment used for
its paper (copied here; that repository stays frozen as a paper record, checked out at `$VLLM_EXL3` when its reference
runs are needed). Every copied file comes
from that repository at commit 40202fd (40202fd6cb59907ec22bc749cd1ce52244f18bec, 2026-10-06). The whole files
(the two scripts, the subsets, the price page) entered Strata byte for byte in one commit; later commits change them
and say why. `quality.py` takes four functions of a module of that repository, with the one change below.

| file here | source at 40202fd | changes in Strata |
|---|---|---|
| `run_lm_eval.sh` | `quality/run_lm_eval.sh` | see its header: no `stack/pins.env` (lm-eval 0.4.13 pinned in the script) |
| `api_proxy.py` | `quality/api_proxy.py` | the `llamacpp` and `strata` backends, `--key-file`, `--api-base`; `deepseek` and `vllm` unchanged |
| `subsets_lean2h_seed20261006.json` | `quality/subsets_lean2h_seed20261006.json` | none |
| `deepseek-api-pricing-2026-10-06.html` | `data/static/pricing/deepseek-api-pricing-2026-10-06.html` | none (the price page the 2026-10-06 API run was priced with) |
| `quality.py` | `src/hostoffload/analysis/quality.py` (`find_results`, `headline`, `load_samples`, `paired_difference`) and `stats.mean` of `src/hostoffload/analysis/stats.py` | `load_samples` takes the filter (gsm8k logs one line per filter) |

The pinned method (unchanged from the source): lm-evaluation-harness 0.4.13, seeds `1234,1234,1234,1234`, temperature
0, thinking off, `--log_samples`, `--apply_chat_template`, per-task `max_gen_toks` (gsm8k_cot_llama 1024, ifeval 1280,
humaneval_instruct 1024), the fixed-seed subsets above (GSM8K CoT 150 items, IFEval 150, HumanEval instruct 164), and
the paired percentile bootstrap (5000 resamples, seed 0) of `paired_difference`.

The reference runs the analysis reads (`analyze.py`) stay in that repository (paths under `$VLLM_EXL3`):
`data/controlled/E11-quality-reference/api/20261006T181137Z` (the DeepSeek API, 2026-10-06) and
`data/controlled/E10-quality-exl3/exl3-offload-gen30-lean-v2/20261006T211059Z` (the EXL3 3 bpw stack).
