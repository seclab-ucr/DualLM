#!/bin/bash
#set up the environment. install conda before running this script..https://www.anaconda.com/download/success

#setup the environment for running the step llm query part.
conda create --name dl_step1 python=3.10
conda activate dl_step1
#install the dependencies
conda install -c conda-forge openjdk=21
pip3 install -r requirements.txt

conda deactivate
#setup the environment for running the step 2 part.

conda create --name dl_step2 python=3.6.10
conda activate dl_step2
#install the dependencies
conda config --add channels conda-forge
conda config --add channels pytorch

conda install pytorch==1.5.1 torchvision==0.6.1 cudatoolkit=10.2 -c pytorch
conda install submitit sklearn

pip3 install -r requirements2.txt