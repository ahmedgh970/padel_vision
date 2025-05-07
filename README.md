# Project Structure
```text
padel_stats/                  # Root project directory
├─ src/                       # Source code directory
│  ├─ padel_detection/        # Detection modules package
│  │  ├─ __init__.py          # Expose Detector and process_video
│  │  ├─ detector.py          # Pose & ball detection wrappers
│  │  ├─ video_processor.py   # Video frame processing logic
│  │  └─ main.py              # CLI for detection
│  └─ video_preprocessor/     # Video preprocessing package
│     ├─ __init__.py          # Expose preprocessing functions
│     ├─ preprocessor.py      # Frame sampling logic
│     └─ main.py              # CLI for preprocessing
├─ tests/                     # Unit tests
│  ├─ test_detector.py
│  └─ test_video_processor.py
├─ setup.py                   # Package installation and entry points
├─ requirements.txt           # Python dependencies
├─ README.md                  # Overview and usage
└─ .gitignore                 # Git ignore rules
```