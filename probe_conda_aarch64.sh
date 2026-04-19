# Check env's libstdc++ version, then force it ahead of any module-loaded libstdc++.

ls -la /net/storage/pr3/plgrid/plggllmeffi3/nano/plgj321m/pixi/.pixi/envs/default/lib/libstdc++.so.6*

strings /net/storage/pr3/plgrid/plggllmeffi3/nano/plgj321m/pixi/.pixi/envs/default/lib/libstdc++.so.6 | grep -E 'CXXABI_1\.3\.1[0-9]' | sort -u | tail

export LD_LIBRARY_PATH=/net/storage/pr3/plgrid/plggllmeffi3/nano/plgj321m/pixi/.pixi/envs/default/lib:$LD_LIBRARY_PATH

pixi run python -c "import torch, triton; print(torch.__version__, triton.__version__)"
