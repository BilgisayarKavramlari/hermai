#!/bin/bash
# Experiments added in the revision: established metrics on the stored explanations, and ROAR.
#   sh run_revision_jobs.sh baselines
#   sh run_revision_jobs.sh roar
# If lime 0.2.0.1 is not installed, set LIME_SRC to the folder of its source distribution (see README).
[ -n "$LIME_SRC" ] && export PYTHONPATH="$LIME_SRC:$PYTHONPATH"
cd "$(dirname "$0")"
if [ "$1" = "baselines" ]; then
  for ds in pima compas breast adult; do for mk in rf mlp; do python3 baselines.py $ds $mk 0; done; done
elif [ "$1" = "roar" ]; then
  for ds in pima breast; do for mk in rf mlp; do python3 roar.py $ds $mk 0; done; done
fi
