#!/bin/bash

set -eou pipefail

git clone git@github.com:NickCody/mydream.git
cd mydream
git clone --branch 2025-02-08-nickcody git@github.com:NickCody/CodeFormer.git