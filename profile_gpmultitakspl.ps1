$env:PYTHONPATH += "./factorio"
python -m cProfile -o .out/planner.prof factorio/gpmodels/gpmultitaskpl.py
snakeviz .out/planner.prof