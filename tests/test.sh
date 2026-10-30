#!/bin/bash
set -uo pipefail

mkdir -p /logs/verifier
python3 /tests/grade.py
status=$?

# If the grader was SIGKILL'd (OOM) or otherwise failed before writing a reward,
# Harbor needs a file; score that attempt as 0.
if [ ! -f /logs/verifier/reward.txt ]; then
    echo "0.000000" > /logs/verifier/reward.txt
    echo '{"reward": 0.0}' > /logs/verifier/reward.json
fi

if [ "$status" -ne 0 ]; then
    exit "$status"
fi

pytest --ctrf /logs/verifier/ctrf.json /tests/test_state.py -q
exit 0
