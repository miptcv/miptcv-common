#!/usr/bin/env bash
conda create -n miptcv python=2.7 ipython=4.2.0 opencv matplotlib

conda install opencv

source activate miptcv
source deactivate
