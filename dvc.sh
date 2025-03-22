FAIL=$(dvc repro "$@" 2>&1 && dvc push 2>&1) || python scripts/notify.py "$FAIL" idev
