# DualLM

**DualLM** is an open-source tool for detecting security-critical Linux kernel patches, focusing on **use-after-free (UAF)** and **out-of-bounds (OOB)** vulnerabilities. It uses a dual-model pipeline combining a **Large Language Model (LLM)** and a **fine-tuned lightweight model** to analyze commit messages and code diffs with high accuracy.

📌 **Highlights**:
- 87.4% accuracy, 0.875 F1-score  
- Outperforms SID and TreeVul  
- Identified 90+ verified UAF/OOB patches (incl. silent fixes)  
- Includes PoCs for confirmed vulnerabilities  

DualLM helps downstream maintainers and security teams prioritize critical patches faster and more reliably.

# Setup

## Hardware Requirements

- Ubuntu Desktop (recommended: Ubuntu 24.04 LTS)
- NVIDIA GPU with CUDA support
- Minimum 16GB RAM recommended


## Environment
We use conda to manage the running environemnt, please see https://anaconda.org/anaconda/conda for details to install conda.

## Dependencies

DualLM relies on Joern(https://github.com/joernio/joern) for the slicing.  A jern.zip has been uploaded to the root folder of DualLMs. please unzip it and keep it under DualLMs root.

## Models

The models are stored under PROJECT_FOLDER/models.  just download them and keep them there.

## API Keys

DualLM requires an OpenAI API key to function. To set up your API key: edit dualLM.sh.
```
#export OPENAI_API_KEY="your-openai-api-key-here"
```

## Linux Repo.
clone Linux repo into ./repos/. so the system can fetch the commits information from repo.

## Input & Output
   The input is a list of commits, output is the evaluation of those commits, are they security related patches, etc.

## Run DualLM.
### Setup Environment

``` ./setup.sh
```
### Run DualLM
Modify the following lin in dualLM.sh to include linux commits you wanna analyze.
```
COMMITS="ff2047fb755d d1e7fd6462ca f4020438fab0 2c1f6951a8a8 79dc7e3f1cd3"
```

Run the script.

```
./dualLM.sh
```

This script will 1) run the analysis using LLM. It will identify "reliable" commits", which are are commits that contain enough information to determine if this commit is a security patch and its type. For othe commits (not-reliable commits), their patching purpose cannot be determined at this stage.  Step 2 will conduct further analysis using customzied models with slicing data.  More details can be foudn in dualLM.sh. 

## Results.
Under folder data/results, you can see the detailed resutls.

for unreliabe commits, the system will tell if it is a security patch, and if yes, what type of vulnerability it patched.
