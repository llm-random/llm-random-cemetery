#!/bin/bash
COMMAND=$(command -v apptainer >/dev/null && echo apptainer || echo singularity)
# $COMMAND build --fakeroot sparsity-base.sif sparsity-base.def
$COMMAND build --fakeroot /pfs/lustrep4/scratch/project_465001227/llm-random-group/sparsity_`date +'%Y.%m.%d_%H.%M.%S'`.sif sparsity-head.def