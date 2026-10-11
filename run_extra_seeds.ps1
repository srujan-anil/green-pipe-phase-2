python -m experiments.runner --strategy baseline --zone DE --seed 1 --epochs 3 --max-sim-hours 24
python -m experiments.runner --strategy reactive --zone DE --seed 1 --epochs 3 --max-sim-hours 24
python -m experiments.runner --strategy predictive --zone DE --seed 1 --epochs 3 --max-sim-hours 24
python -m experiments.runner --strategy adaptive --zone DE --seed 1 --epochs 3 --max-sim-hours 24
python -m experiments.runner --strategy multiregion --zone DE --seed 1 --epochs 3 --max-sim-hours 24
python -m experiments.runner --strategy green_pipe --zone DE --seed 1 --epochs 3 --max-sim-hours 24

python -m experiments.runner --strategy predictive --zone DE --seed 2 --epochs 3 --max-sim-hours 24
python -m experiments.runner --strategy adaptive --zone DE --seed 2 --epochs 3 --max-sim-hours 24
python -m experiments.runner --strategy multiregion --zone DE --seed 2 --epochs 3 --max-sim-hours 24
python -m experiments.runner --strategy green_pipe --zone DE --seed 2 --epochs 3 --max-sim-hours 24
python -m experiments.runner --strategy baseline --zone DE --seed 2 --epochs 3 --max-sim-hours 24
python -m experiments.runner --strategy reactive --zone DE --seed 2 --epochs 3 --max-sim-hours 24

python -m experiments.runner --strategy multiregion --zone DE --seed 3 --epochs 3 --max-sim-hours 24
python -m experiments.runner --strategy green_pipe --zone DE --seed 3 --epochs 3 --max-sim-hours 24
python -m experiments.runner --strategy baseline --zone DE --seed 3 --epochs 3 --max-sim-hours 24
python -m experiments.runner --strategy reactive --zone DE --seed 3 --epochs 3 --max-sim-hours 24
python -m experiments.runner --strategy predictive --zone DE --seed 3 --epochs 3 --max-sim-hours 24
python -m experiments.runner --strategy adaptive --zone DE --seed 3 --epochs 3 --max-sim-hours 24

python -m experiments.aggregate