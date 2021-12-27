import librosa
import numpy as np
from os import listdir
from os.path import isfile, join

def load_wav_files(data_path):
    """Loads files from given directory into dicitonary

    Files are stored in a dictonary, where keys are speakers ids and values 
    are lists of audio time sereis, so that each index in a list coresponds 
    to the digit spoken by the speaker.

    Parameters
    ----------
    data_path : str
        path to directory with training data

    Returns
    -------
    speaker_files : dict of {str : list of np.ndarray [shape=(n,)]}
    """
    SPEAKER_ID_LEN = 5

    wav_filenames = [filename for filename in listdir(data_path) if isfile(join(data_path, filename)) and filename.endswith('.wav')]
    wav_filenames.sort()

    speaker_ids = set()
    for filename in wav_filenames:
        speaker_ids.add(filename[0:SPEAKER_ID_LEN])
    speaker_ids = sorted(speaker_ids)

    speaker_files = {id_ : [] for id_ in speaker_ids}
    for filename in wav_filenames:
        id_ = filename[0:SPEAKER_ID_LEN]
        x, fs = librosa.load(join(data_path, filename), sr=None)
        speaker_files[id_].append(x)

    return speaker_files



def cmvn(mfcc):
    """Cepstral mean and variance normalization (CMVN)

    Parameters
    ----------
    mfcc : np.ndarray [shape=(n_mfcc, t)]
        MFCCs matrix
    
    Returns
    -------
    cmvn : np.ndarray [shape=(n_mfcc, t)]
    """
    stdevs = np.std(mfcc,1)
    means = np.mean(mfcc,1)
    cmvn = ((mfcc.T - means)/stdevs).transpose()
    return cmvn



def generate_speaker_mfccs(speaker_files, cmv_norm=False, deltas=False, **kwargs):
    """Generates Mel-frequency cepstral coefficients (MFCCs)

    Singlas in form of MFCCs are stored in the same way as in `load_wav_files`.
    It also removes the #0 coefficient.

    Parameters
    ----------
    speaker_files : dict of {str : list of np.ndarray [shape=(n,)]}
        dictionary of files assigned to speaker id
    cmv_norm : bool, deafult False
        cepstral mean and variance normalization
    deltas : bool, deafult False
        compute local estimate of the derivative of the mfcc data
    kwargs : additional keyword arguments
        Arguments to `librosa.feature.mfcc`

    Returns
    -------
    speaker_mfccs : dict of {str : list of np.ndarray [shape=(t, n_mfcc-1) or shape=(t, 3*(n_mfcc-1))]}
    """
    speaker_mfccs = {id_ : [] for id_ in speaker_files}
    
    for id_ in speaker_files:
        for digit_wav in speaker_files[id_]:
            mfcc = librosa.feature.mfcc(y=digit_wav, **kwargs)[1:]
            
            if cmv_norm == True:
                mfcc = cmvn(mfcc)

            if deltas == True:
                mfcc_delta = librosa.feature.delta(mfcc)
                mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
                mfcc = np.concatenate((mfcc, mfcc_delta, mfcc_delta2), axis=0)
                
            speaker_mfccs[id_].append(mfcc.T)

    return speaker_mfccs



def generate_digits_mfccs(speaker_mfccs):
    """Concatenates MFCCs from all speakers into signle array

    MFCCs are stored in a list, so that each index in the list coresponds
    to a digit spoken by a speaker.

    Parameters
    ----------
    speaker_mfccs : dict of {str : list of np.ndarray [shape=(t_1, n_mfcc-1) or shape=(t_1, 3*(n_mfcc-1))]}
        dictionary of MFCCs assigned to speaker id
    
    Returns
    -------
    digits_mfccs : list of np.ndarray [shape=(t_2, n_mfcc-1) or shape=(t_2, 3*(n_mfcc-1))]
    """
    digits_mfccs = []

    for n in range(10):
        digits_mfccs.append(np.concatenate([speaker_mfccs[id_][n] for id_ in speaker_mfccs]))

    return digits_mfccs