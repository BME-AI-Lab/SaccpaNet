# Experiments

## Index
The experiments were run in the following sequences
1. Random Search
2. Manual Search
3. Pretrainning
4. Weight Transfers 
5. Finetuning
6. Posture Classification training 


## Random Search
``` batch
cd 01-Random_search
mkdir runs
python generate_runs.py
python -m helper.local_run -f runs -s search.py
```