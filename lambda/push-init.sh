#!/bin/bash

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Get our shell and helpers (~/dotfiles)
#
if declare -F sshdotfiles  > /dev/null; then
    sshdotfiles lambda || true
fi

envsubst < $PROJECT_ROOT/lambda.localrc | scp - lambda:.localrc

# Clone the repo
#
ssh lambda "git clone git@github.com:NickCody/mydream.git"

# Setup HuggingFace stuff
#
ssh lambda "mkdir HuggingFace 2> /dev/null"
rsync -avz $SAFETENSOR_HOME/lambda/*.safetensors lambda:HuggingFace

