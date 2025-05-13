# Padel Vision
![padel Vision](./assets/)

# Project Structure
```text
padel_vision/                 # Root project directory
├─ src/                       # Source code directory
│  ├─ padel_detect/           # Package for keypoints and ball detection
│  │  ├─ __init__.py          
│  │  ├─ detector.py          
│  │  ├─ detection_pipe.py    
│  │  └─ main.py              
│  ├─ padel_preprocess/       # Package for video preprocessing
│  │  ├─ __init__.py          
│  │  ├─ preprocessor.py      
│  │  └─ main.py              
│  └─ padel_pipe/             # Package for sampling, detection, export, annotating video
│    ├─ __init__.py          
│    ├─ annotate.py 
│    └─ main.py        
├─ tests/                     # Unit tests
│  ├─ test_detector.py
│  ├─ test_detection_pipe.py         
│  ├─ test_preprocessor.py    
│  └─ test_annotate.py 
├─ setup.py                   # Package installation and entry points
├─ README.md                  # Overview and usage
└─ .gitignore                 # Git ignore rules
```

# Setup
#### 1. Clone this repository.
#### 2. Setup virtual environment.
#### 3. Install packages in editable mode.
```
conda create -n python=3.10 padelvis
conda activate padelvis
pip install -e .
```

# Inference
At the root of this repo, edit the config files configs/ accordingly and run:
````
padel-preprocess --config-dir configs/   #-- for video preprocessing 
padel-detect --config-dir configs/       #-- for keypoints and ball detection from video/image
padel-pipe --config-dir configs/         #-- for the full pipeline (prep detect plot)
````