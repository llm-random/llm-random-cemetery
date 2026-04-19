# Nuke the corrupt env and reinstall from scratch.

rm -rf /net/storage/pr3/plgrid/plggllmeffi3/nano/plgj321m/pixi/.pixi/envs/default

pixi install

pixi list | grep -iE 'triton|torch'

pixi run python -c "import importlib.util; print(importlib.util.find_spec('triton'))"

pixi run python -c "import torch, triton; print(torch.__version__, triton.__version__)"
