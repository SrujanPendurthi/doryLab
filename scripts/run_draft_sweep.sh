#!/usr/bin/env bash
cd "$(dirname "$0")/.."
until grep -q FOLLOWUPS_DONE results/logs/followups.log || ! pgrep -f run_followups.sh >/dev/null; do sleep 60; done
PYTHONPATH=. .venv/bin/python scripts/draft_sweep.py
echo DRAFT_SWEEP_DONE
