# Get started
In this section we demonstrate how to preapre the environment for the experiment. 
The code is tested on windows, and in theory should also work on linux. 

## Installing related dependecies
### Using Windows with conda
Create the `pytorch-mamba` python environment based and conda and the provided enviroment.yml
``` batch
conda env create -f environment.yml
condfa activate posture_experiment
```
### Building the environment from scratch with conda and pip
Please be aware that pytorch-lightning and sqlsalchemy are setted to specific version as newer versions introduces breaking changes. 
``` batch
mamba create -n posture_experiment -c pytorch -c nvidia -c conda-forge -c default python=3.9 pytorch torchvision torchaudio pytorch-cuda pandas seaborn -y
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