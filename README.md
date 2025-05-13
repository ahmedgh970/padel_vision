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