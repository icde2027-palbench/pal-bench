# Full50 Official Evaluation Report

Run ID: `pal_bench_icde2027_20260528_195431`

## Scope

- Users: 50 (`user_0000` through `user_0049`)
- Agent LLM role: `agent_llm` (`gpt-5.4`)
- Official judge role: `eval_judge` (`qwen3.6-35b-a3b`)
- Judge modes: `judge-mode=llm_semantic`, `evidence-judge-mode=llm_semantic`

## Generation Status

| Method | Profiles | LLM calls | Parse failures |
|---|---:|---:|---:|
| no_llm_heuristic | 50 | 0 | 0 |
| text_only_profile | 50 | 50 | 0 |
| multimodal_rag | 50 | 50 | 0 |
| long_context_mm_llm | 50 | 50 | 0 |
| generic_tool_agent | 50 | 105 | 5 |
| adapted_prior_lifelog | 50 | 50 | 0 |
| pal_trace | 50 | 372 | 0 |

## Official Full50 Metrics

| Method | OFR | PIR | PIR-hard | PRR-ID | EFS | ECE | nLLM |
|---|---:|---:|---:|---:|---:|---:|---:|
| no_llm_heuristic | 0.3799 | 0.4047 | 0.1756 | 0.5749 | 0.2306 | 0.3749 | 0.0000 |
| text_only_profile | 0.4041 | 0.0000 | 0.0000 | NA | 0.1088 | 0.0161 | 1.0000 |
| multimodal_rag | 0.3580 | 0.2942 | 0.1573 | 0.9175 | 0.2213 | 0.1116 | 1.0000 |
| long_context_mm_llm | 0.4009 | 0.2336 | 0.1617 | 0.8446 | 0.1968 | 0.1075 | 1.0000 |
| generic_tool_agent | 0.2560 | 0.3321 | 0.1405 | 0.8352 | 0.2039 | 0.1628 | 2.1000 |
| adapted_prior_lifelog | 0.3921 | 0.3900 | 0.2098 | 0.8328 | 0.2741 | 0.1528 | 1.0000 |
| pal_trace | 0.6057 | 0.4792 | 0.2684 | 0.8384 | 0.3764 | 0.3236 | 7.4400 |

## Bootstrap Highlights

Paired user-level bootstrap, 10,000 resamples. Deltas are `pal_trace - baseline`.

| Baseline | OFR delta [95% CI] | PIR delta [95% CI] | EFS delta [95% CI] |
|---|---:|---:|---:|
| long_context_mm_llm | 0.2048 [0.1637, 0.2452] | 0.2456 [0.1811, 0.3117] | 0.1795 [0.1517, 0.2099] |
| generic_tool_agent | 0.3497 [0.3125, 0.3857] | 0.1471 [0.0888, 0.2074] | 0.1724 [0.1421, 0.2038] |
| multimodal_rag | 0.2478 [0.2087, 0.2842] | 0.1850 [0.1109, 0.2586] | 0.1550 [0.1233, 0.1886] |
| adapted_prior_lifelog | 0.2137 [0.1793, 0.2485] | 0.0892 [0.0186, 0.1598] | 0.1022 [0.0696, 0.1372] |
| no_llm_heuristic | 0.2259 [0.1897, 0.2633] | 0.0745 [0.0375, 0.1147] | 0.1457 [0.1225, 0.1711] |

PIR-hard is directionally positive for all baselines but should be reported with nuance for long-context because its 95% CI crosses zero.

## Key Artifacts

- Main table: `tables/official_llm_semantic_full50/main_multisystem_results.md`
- Diagnostics export: `diagnostics/official_llm_semantic_full50`
- Bootstrap summary: `diagnostics/bootstrap_official_full50/bootstrap_summary.md`
- Per-method official aggregates: `eval/*/official_llm_semantic/aggregate_formal.json`
- Commands manifest: `manifests/commands.sh`
- Full run logs: `logs/*_full50*.log`

## Notes

- Official full50 evaluation completed without judge JSON interruptions.
- A value-judge JSON retry/repair fix was added before the full pass and covered by unit tests.
- `generic_tool_agent` required 5 parse retries during generation but all users succeeded.
