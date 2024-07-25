# SaccpaNet: A Separable Atrous Convolution-based Cascade Pyramid Attention Network to Estimate Body Landmarks Using Cross-modal Knowledge Transfer for Under-blanket Sleep Posture Classification 

[Documentation](http://www.ai-materials-lab.net/SaccpaNet/docs/build/html/index.html)


# Get started
In this section we demonstrate how to preapre the environment for the experiment. 
The code is tested on windows, and in theory should also work on linux. 

# Data and checkpoints
The data and checkpoints can be downloaded from [onedrive](https://connectpolyu-my.sharepoint.com/:f:/g/personal/20106983r_connect_polyu_hk/EtIovZKxnyRDne8C1e1QTYIB7zYpKYPiyEH6SfBf2wGeTQ?e=SdTZqB)

## Installing related dependecies
### Using Windows with conda
Create the `posture_experiment` python environment based and conda and the provided enviroment.yml
``` batch
conda env create -f environment.yml
conda activate posture_experiment
```
### Building the environment from scratch with conda and pip
Please be aware that pytorch-lightning and sqlsalchemy are setted to specific version as newer versions introduces breaking changes. 
``` batch
conda create -n posture_experiment -c pytorch -c nvidia -c conda-forge -c default python=3.9 pytorch torchvision torchaudio pytorch-cuda pandas seaborn -y
conda activate posture_experiment
pip install pytorch-lightning==1.6.4 SQLAlchemy==1.4.48 opencv-python scikit-learn tqdm torchmetrics
```


## Install the local project
This provides the networks, data set, and configs shared by the expriment, which can be directly edited,

``` batch
pip install -e . 
```

## installing dependencies for mmpose
``` batch
pip install -U openmim
mim install mmengine
mim install "mmcv>-2.0.0"
mim install "mmpose>=1.0.0"
```

## Additional dependecies for building the documentation
``` batch
pip install sphinx myst-parser sphinx-copybutton sphinx-markdown-tables sphinx-rtd-theme
```

## Additional dependenceis for building tests
``` batch
pip install coverage pytest xdoctest
```


# Project organization
This project is organized with *Experiments* and *a single source of networks*.

The networks and libraries are located at `src`, Experiments are located at root, starting with two digit number.

## Dependencies


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
cd 05-Finetuning
python finetune_parameterized_by_config.py
```
It will produce two weight file in `05-Finetuning\log\SACCPA_sample\05-Finetuning\lightning_logs\version_0\checkpoints`
Copy the file `best-epoch=*-val_loss=*.ckpt` to file `06-Posture_Classification\runs\best-epoch.ckpt`  .

## 06. Posture Classification
``` batch
:: Generate the runs for all classification models being tried.
cd 06-Posture_Classification
python generate_runs.py
:: Running the the list of classification models
python -m helper.local_run -f runs -s finetune_classification.py

```

## Additional experiments (evaluations)
7. Additional Network - transformers
8. flop and params calculations
9. Ablation studies (JCE Saccpa attention block network structures)
10. Ablation studies (JCE Saccpa attention block hyperparameters)
11. Ablation studies (SPC Saccpa attention block network structures)
12. SVMs
13. (Not available) Coordinate Visualizations
14. post-hoc data augmentation robustness test (Ph-DART)
15. ROC ploting

