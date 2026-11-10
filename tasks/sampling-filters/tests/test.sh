#!/bin/bash
# Writes 1 to the reward file only if every hidden test passes.
mkdir -p /logs/verifier
if APP_DIR=/app python -m pytest /tests -q -rA -p no:cacheprovider; then
  echo 1 > /logs/verifier/reward.txt
else
  echo 0 > /logs/verifier/reward.txt
fi
