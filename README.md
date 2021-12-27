# Digits recognition
> This is ready to use API for training, tuning, evaluating Gaussian Mixture Model used for spoken digits recognition. We also provided notebook, that presents achieved results.

## Features
API provides functions for: 
- Hyperparameters optimization like: MFCC features, MFCC deltas, GMM parameters
- Function for crossvalidation
- Function for training GMM model for each number
- Function for evaluation

## Usage
We presented usage of this interface in demo.ipynb. It loads all needed packages. 

We do not provide the training data nor the evaluation data. If you wish to train this model using your data, it needs to be in one folder, each recording has to be named like: `AO1M1_0_.wav`, where first 5 letters are speaker id, and the second part indicates the digit. 

## Results:
The best scores have been achieved for:

##### GMM parameters:
- number of components: 10
- covariance type: diagonal
##### MFCC parameters:
- number of MFCC coefficients: 13
- number of fft samples: 512
- hop length: 160
- window length: 320
- window: hamming
- number of mel filters: 16

The model was trained on data from only male speakers. 

Accuracy achieved on validation data (both male and female speakers from the same dataset - same recording conditions): 93.50%
Accuracy achieved on unknown data (data was recorded by various people - various recording conditions, often poor quality): 70.73%


## Technologies used
- librosa
- numpy
- os
- matplotlib
- sklearn
- csv
- jupyter-notebook

## Contact
Created by lukasz-k1, koksik327, thatguyzz, PhilPhedora - feel free to contact us!
