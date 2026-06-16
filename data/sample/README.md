# Sample Data

This directory contains one synthetic PAL-Bench user for smoke tests and schema
inspection. It is not the full benchmark.

Use it with:

```bash
python scripts/evidence_chain_dossier/run_single.py \
  --user user_0000 \
  --users-root data/sample/users \
  --output-root outputs/sample_pal_trace \
  --force-no-llm
```
