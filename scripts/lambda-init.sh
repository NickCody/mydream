#!/bin/bash

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

sshdotfiles lambda || true
scp $PROJECT_ROOT/lambda.localrc lambda:.localrc
ssh lambda "mkdir HuggingFace 2> /dev/null"
rsync -avz $SAFETENSOR_HOME/lambda/*.safetensors lambda:HuggingFace

