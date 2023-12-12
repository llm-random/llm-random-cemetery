# setup git hooks
git config --local core.hooksPath .githooks

# is the user using conda?
if [ -n "$CONDA_DEFAULT_ENV" ]; then
    CONDA_ENV="llm-random"
    echo "[start-dev.sh] You're using conda - will create a conda environment"
    conda create --name $CONDA_ENV python=3.10
    conda activate $CONDA_ENV
    $SHELL install_requirements.sh
    fi
else
    # setup the virtual environment
    if [ ! -d "venv" ]; then
        python3 -m venv venv
        . venv/bin/activate
        ./install_requirements.sh
    else
        echo "[start-dev.sh] I don't know Walt, seems kind of sus to me. You've already got a venv folder, do you really want to make another one? Try again."
    fi
fi