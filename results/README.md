# Results

This directory contains lightweight paper-facing artifacts copied from the
official full50 experiment run:

- `paper_tables/full50_official_report.md`
- `manifests/final_result_manifest.json`

The report is intended for quick inspection of the paper tables. The manifest
records provenance for the official full-run bundle; its paths are public
release paths or relative paths inside the original experiment bundle, not a
promise that every intermediate run directory is stored in git.

The full 50-user JSON benchmark artifact is documented in the repository
README. Large run directories, logs, judge caches, and per-user prediction
outputs are intentionally kept out of git and should be attached as separate
release artifacts if exact offline regeneration of every paper table is needed.
