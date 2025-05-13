from setuptools import setup, find_packages

setup(
    name='padel_shot',
    version='0.0.1',
    description='Padel shot detection and statistics extraction',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'torch',
        'torchvision',
        'ultralytics',
        'opencv-python',
        'numpy',
        'pytest',
        'opencv-python',
        'hydra-core',
        'omegaconf',
    ],
    entry_points={
        'console_scripts': [
            'padel-detect=padel_detect.main:cli',
            'padel-preprocess=padel_preprocess.main:cli',
            'padel-pipe=padel_pipe.main:cli'
        ],
    },
)