#!/bin/bash

. load_config.sh

usage="Syntax: $(basename "$0") [-h|--help] [COMPONENTS...] -- will set up the project environment,

where:
    -h | --help     Print this help
    COMPONENTS...   Specify component to set up

Valid COMPONENTS:
    all: set up all components (container will be pulled, not built)
    cont_[pull|build]: pull the singularity container or build it
    python: install the python environment via poetry
    data: pull data"

if [[ $# -eq 0 ]] || [[ "$@" =~ "--help" ]] || [[ "$@" =~ "-h" ]];then
    echo "$usage"
    exit 0
fi

# container setup
if [[ "$@" =~ "cont_pull" ]] || [[ "$@" =~ "all" ]];then
    echo "Pulling singularity container..."
    wget "https://yale.box.com/shared/static/i0fvuupeois4kpmlmhjod4j7bthfliiv.sif" -O "${ENV[cont]}"
elif [[ "$@" =~ "cont_build" ]];then
    echo "Building singularity container..."
    SINGULARITY_TMPDIR=/var/tmp sudo -E singularity build "${ENV[cont]}" Singularity
else
    echo "Not touching container"
fi

# python env setup
if [[ "$@" =~ "python" ]] || [[ "$@" =~ "all" ]];then
    echo "Setting up python environment..."
    singularity exec ${ENV[cont]} bash -c "poetry install"
else
    echo "Not touching python environment"
fi

# download stimulus set
if [[ "$@" =~ "data" ]] || [[ "$@" =~ "all" ]];then
    echo "Pulling data..."
    wget "https://yale.box.com/shared/static/xll19zcbtrxm69cb46b42kjt7rgm6dph.gz" -O "neural-data.tar.gz"
    wget "https://yale.box.com/shared/static/kvm1cywpkforwo5puhb6rzhtbtom2crn.gz" -O "network-activations.tar.gz"
    wget "https://yale.box.com/shared/static/n72fn9ri5narr6tizlwxjvhab5nnq1m6.gz" -O "images.tar.gz"
    wget "https://yale.box.com/shared/static/t563v9d78nf2ni9mjdwd60tpnklxc5q1.gz" -O "network-weights.tar.gz"
    wget "http://visiondata.cis.upenn.edu/spin/data.tar.gz"
    tar -xzf neural-data.tar.gz && rm neural-data.tar.gz
    tar -xzf network-activations.tar.gz && rm network-activations.tar.gz
    tar -xzf images.tar.gz && rm images.tar.gz
    tar -xzf network-weights.tar.gz && rm network-weights.tar.gz
    tar -xvf data.tar.gz &&
    wget http://visiondata.cis.upenn.edu/spin/model_checkpoint.pt --directory-prefix=data && \
    wget "https://yale.box.com/shared/static/334hax8svjinbi7t8sb1plun7fb8et48.pkl" -O "basic_smpl_neutral.pkl" --directory-prefix=data && \
    rm *.tar.gz
else
    echo "Not pulling any data"
fi
