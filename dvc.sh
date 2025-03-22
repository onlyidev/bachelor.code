args=$(printf '%q ' "$@")
dvc repro "$@" && dvc push || python scripts/notify.py "Experiment (args: $args) failed!" idev
