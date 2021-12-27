from data_processing import cmvn
from sklearn import mixture
import numpy as np
import librosa

def train_gmm(mfccs, **kwargs):
    """Trains Gaussian Mixture Model (GMM)

    Parameters
    ----------
    mfccs : np.ndarray [shape=(t, n_mfcc)]
        MFCCs of a digit concatenated from all speakers in training set
    kwargs : additional keyword arguments
        Arguments to `sklearn.mixture.GaussianMixture`

    Returns
    -------
    gmm : `sklearn.mixture.GaussianMixture` object
    """
    gmm = mixture.GaussianMixture(**kwargs)
    gmm.fit(mfccs)
    return gmm



def train_gmms(digits_mfccs, **kwargs):
    """Creates list of GMM objects

    Each index in a list coresponds to a digit spoken by the speakers. 

    Parameters
    ----------
    digits_mfccs : list of np.ndarray [shape=(t, n_mfcc)]
        list of MFCCs assigned to each number
    kwargs : additional keyword arguments
        Arguments to `sklearn.mixture.GaussianMixture`

    Returns
    -------
    gmms : list of `sklearn.mixture.GaussianMixture` objects
    """
    gmms = [train_gmm(mfccs, **kwargs) for mfccs in digits_mfccs]
    return gmms



def caluclate_scores(gmms, mfccs):
    """Calculates log-likelihoods for all models from given mfcc sample

    Parameters
    ----------
    gmms : list of `sklearn.mixture.GaussianMixture` objects
        list of GMM models
    mfcc : np.ndarray [shape=(t, n_mfcc)]
        MFCCs matrix

    Returns
    -------
    scores : list of float
    """
    scores = [model.score(mfccs) for model in gmms]
    return scores



def predict_digit(gmms, mfccs=None, file_path=None, cmv_norm=False, deltas=False, **kwargs):
    """Returns digit with the highest log-likelihood and it's likelihood

    Parameters
    ----------
    gmms : list of `sklearn.mixture.GaussianMixture` objects
        list of GMM models
    mfcc : np.ndarray [shape=(t, n_mfcc)], deafult None
        MFCCs 
    file_path : str, deafult None
        path to file with .wav recording
    cmv_norm : bool, deafult False
        cepstral mean and variance normalization, only if file_path != None
    deltas : bool, deafult False
        compute local estimate of the derivative of the mfcc data, only if file_path != None
    kwargs : additional keyword arguments
        Arguments to `librosa.feature.mfcc`, only if file_path != None 

    Returns
    -------
    predicted_digit : int
    likelihood : float
    """

    if file_path != None:
        x, fs = librosa.load(file_path, sr=None)
        mfcc = librosa.feature.mfcc(y=x, sr=fs, **kwargs)[1:]
        if cmv_norm == True:
            mfcc = cmvn(mfcc)
        if deltas == True:
            mfcc_delta = librosa.feature.delta(mfcc)
            mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
            mfcc = np.concatenate((mfcc, mfcc_delta, mfcc_delta2), axis=0)
        mfccs = mfcc.T

    scores = caluclate_scores(gmms, mfccs)

    predicted_digit = np.argmax(scores)
    likelihood = np.max(scores)
    
    return predicted_digit, likelihood