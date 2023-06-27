# Experiments

## List of experiments
The experiments were run in the following sequences
1. Random Search
2. Manual Search
3. Pretrainning
4. Weight Transfers 
5. Finetuning
6. Posture Classification training 


## 1. Random Search
### Running the search
``` batch
cd 01-Random_search
:: Generate the list of random seeds for search
mkdir runs
python generate_runs.py
:: Running the the list of random search
python -m helper.local_run -f runs -s search.py
```
To make use of multiple gpus, you can split the runs into chunks, put them into individual run folders, and copy back when everything finish.


### Aggregating the results
Each run will generate a log.csv file with a line containing the search parameters and the validation loss result.
The helper script ```aggregate_result.py``` is used to aggregate everything into a big ```.csv``` file for analysis in excel.
``` batch
python aggregate_result.py
```

### Analysis of the result
In excel, create a table with the .csv file, and filter for *good* models, which is defined as a threshold, and then plot for each parameters against the performance. The threshold used in this experiment are provided in the paper. 
Update ```configs.random_seaerch_params.py``` with the best results from this step.

## 2. Manual searching

### Runing the models

``` batch
:: Starting from root repo
cd 02-Manual_Search
:: Generate the list of random seeds for search
mkdir runs
python generate_runs.py
:: Running the the list of random search
python -m helper.local_run -f runs -s search.py
```

### Aggregating the results of manual search
See [Aggregate result from random search](#aggregating-the-results).


### Analysis


## 03. Pretraining

Patch mmpose with our custom data.
``` batch
cd 03-Pretraining
git clone https://github.com/open-mmlab/mmpose.git
rclone copy patched_codes/ mmpose

```
Obtain and prepare the *COCO whole-body dataset* according to [MMPose's instruction](https://mmpose.readthedocs.io/en/latest/dataset_zoo/2d_wholebody_keypoint.html#coco-wholebody). 

## 04. Weight transfer
Copy the saved weight from MMPose checkpoint folder ```{folder}``` to ```04--Weight_transfer```

``` batch
cd 04-Weight_Transfer
:: Generate the checkpoint base 
python generate_checkpoint_template.py
:: Transfer the checkpoint
python transfer_weight.py best_coco-wholebody_AP_epoch_*.pth template_checkpoint.pth merged_model.pth 
```

## 05. Finetuning

``` batch
:: Copy the transfered weight to 05-Finetuning folder 
copy 04-Weight_Transfer/merged_model.pth 05-Finetuning/merged_model.pth
:: Run the training
python finetune_parameterized_by_config.py
```

## 06. Posture Classification
``` batch
:: Copy the joint coordinate model weight to 06-Posture_Classification
copy 05-Finetuning/merged_model.pth 06-Posture_Classification
:: Generate the runs for all classification models being tried.
python 

