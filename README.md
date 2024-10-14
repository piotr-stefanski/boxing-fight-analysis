# Introduction
This repository contains the source code for experimental research conducted on the analysis of boxing fight recordings. 
The research focuses on video data, using the publicly available dataset [Olympic Boxing Punch Classification Video Dataset](https://www.kaggle.com/datasets/piotrstefaskiue/olympic-boxing-punch-classification-video-dataset) from Kaggle.

The primary goal of this repository is to ensure the reproducibility of the experiments and findings presented in published papers by the authors.
By providing access to the code and methodologies used, this repository aims to assist researchers, data scientists, and sports analysts in replicating the results, further enhancing the study of boxing video analysis.

# Experiments
## Detecting boxers in the boxing ring
TODO: some text about this stage 

## Detecting Clashes in Boxing
TODO: some text about this stage 

## Punch Classification
TODO: some text about this stage 

## Punch Detecting
TODO: some text about this stage 

## Improve Punch Detecting by video frame segmentation
TODO: some text about this stage 

### Own segmentation approach that reduces processing time
TODO: some text about this stage 

# Reproducing experiments
## Environment preparation
### Requirements
1. Python (tested on Python 3.12.3)
2. pip (tested on pip 23.2.1)
3. Some video to test e.g. `kam4/GH079681.MP4` from [published dataset](TODO) saved into `./data/videos`
4. Pretrained weights to network (tested on ssd mobilenet v3 trained on coco dataset, files available on [Google Drive](https://drive.google.com/drive/folders/1TB3rL7pTCSQhcGYloc-jTIk1l_GvwEeE?usp=sharing)) saved into `./data/models`

### Installation
1. Install dependencies `pip install -r requirements.txt`

## Detecting boxers in the boxing ring
To reproduce run `python detecting_boxers_in_the_boxing_ring.py`

## Detecting Clashes in Boxing
To reproduce run `python detecting_clashes_in_boxing.py`

## Punch Classification
TODO: add command to reproduce

## Punch Detecting
TODO: add command to reproduce

## Improve Punch Detecting by video frame segmentation
TODO: add command to reproduce

### Own segmentation approach that reduces processing time
TODO: add command to reproduce