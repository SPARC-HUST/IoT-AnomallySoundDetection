# Environment Sound Collection & Analysis Platform (ESCA)

ESCA has represented the real-time abnormal environment sound detection.




## Contents

1. [Requirements](#-Requirements)
2. [Installation](#-Installation)
3. [Features](#-Features)
4. [Prepare data and Training](#-Prepare-data-and-Training)
2. [Results](#-Results)
## Requirements
* Python 3.7
* Tensorflow & Tensorflow-GPU version 2.7.0
* Gammatone https://github.com/detly/gammatone


## Installation

Install project with command
```bash
  git clone https://github.com/nqthinh493/Environment-Sound-Collection-Analysis-Platform
  pip install -r requirements.txt
```
    
## Features

### Main Functionality
* **Base Training:** Training model with labelled environment audio dataset
* **Transfer Learning:** Use pre-trained model for update model with new audio dataset. The new model contains the features of both the old and the new dataset
* **Real-time Dectection:** Model implementation to detect anomalies in audio tracks

**Input:** Audio recording files as format.wav

**Output:** Loss value, Normal/Abnormal classificaion of files splitted from the original audio file.



## Prepare data and Training

### Prepare
* Change specific PATH in ```config\params.yaml```

```bash
  TFRECORD: ../temp
  NORMAL: [normal-folder-path]
  ABNORMAL: [abnormal-folder-path]

```


* The model expect audio data length is 2 seconds in TFRecord type:
```bash
  python ./tools/prepare_data.py -cfg ./config/params.yaml
```
### Training
*To run base-training after prepare data: 
```bash
  python ./tools/base_training.py -cfg ./config/params.yaml
```
*To run transferlearing: 
```bash
  python ./tools/tl_training.py -cfg ./config/params.yaml
```
## Results
## Documentation

[Documentation](https://linktodocumentation)
