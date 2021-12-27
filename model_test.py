from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from os.path import isfile, join
from os import listdir
import numpy as np
import csv

import data_processing
import model_train

def cross_validate(X, n_splits, **kwargs):
    """Performs cross validaton accuracy test on given dicitonary of speakers

    For example: if n of speakers = 10 and n_splits = 5, this function will 
    create 5 gropus of 2 speakers, where each group in different iteraton 
    will be used as part of a training set or as a test set.

    Parameters
    ----------
    X : dict of {str : list of np.ndarray [shape=(t, n_mfcc)]
        dictionary of MFCCs assigned to speaker id
    n_splits : int
        number of groups, that X will be divided to
    kwargs : additional keyword arguments
        Arguments to `sklearn.mixture.GaussianMixture`

    Returns
    -------
    mean_acc : float
    stdev_acc : float
    """
    kf = KFold(n_splits=n_splits)
    id_list = np.array(list(X.keys()))

    accuracy = []

    for train_index, test_index in kf.split(X):
        X_train = {id_ : X[id_] for id_ in id_list[train_index]}
        X_test = {id_ : X[id_] for id_ in id_list[test_index]}

        X_train = data_processing.generate_digits_mfccs(X_train)

        gmms = model_train.train_gmms(X_train, **kwargs)

        y_pred = []
        y_true = []

        for id_ in X_test:
            for n, mfccs in enumerate(X_test[id_]):
                y_pred.append(model_train.predict_digit(gmms, mfccs=mfccs)[0])
                y_true.append(n)

        iteration_accuracy = accuracy_score(y_true, y_pred)
        accuracy.append(iteration_accuracy)

    mean_acc = np.mean(accuracy)
    stdev_acc = np.std(accuracy)

    return mean_acc, stdev_acc



def test_gmm_params(speaker_files, n_splits, n_components_range, covariance_types, cmv_norm=False, **kwargs):
    """Test parameters of `sklearn.mixture.GaussianMixture`

    This function generates plot with results.

    Parameters
    ----------
    speaker_files : dict of {str : list of np.ndarray [shape=(n,)]}
        dictionary of files assigned to speaker id
    n_splits : int
        number of groups, that traing set will be divided to
    n_components_range : list of int
        list of n_componets to test
    covariance_types : list of str
        list of covariance types to test
    cmv_norm : bool, deafult False
        cepstral mean and variance normalization
    kwargs : additional keyword arguments
        Arguments to `librosa.feature.mfcc`

    Returns
    -------
    best_gmm_params : dict of {str : int or str}
    """
    speaker_mfccs = data_processing.generate_speaker_mfccs(speaker_files, cmv_norm=cmv_norm, deltas=False, **kwargs)

    fig, ax = plt.subplots(figsize=(9,5))

    beast_mean = -1
    for covariance_type in covariance_types:
        accuracy = []
        accuracy_err = []
        for n_components in n_components_range:
            gmm_arguments = {'n_components' : n_components, 'covariance_type': covariance_type}
            mean_acc, stdev_acc = cross_validate(speaker_mfccs, n_splits, **gmm_arguments)
            accuracy.append(mean_acc)
            accuracy_err.append(stdev_acc)
            if mean_acc > beast_mean:
                beast_mean = mean_acc
                best_gmm_params = gmm_arguments
        ax.errorbar(n_components_range, accuracy, yerr=accuracy_err, label=covariance_type)
    
    ax.set_xlabel('N. of components')
    ax.set_ylabel('Accuracy')
    ax.set_title('GMM parameters')
    ax.set_xticks(n_components_range)
    ax.set_ylim((0, 1))
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    plt.show()

    return best_gmm_params



def test_mfcc_deltas(speaker_files, n_splits, cmv_norm=False, **kwargs):
    """Tests wheter the derivatives of MFCCs will improve the results
    
    Parameters
    ----------
    speaker_files : dict of {str : list of np.ndarray [shape=(n,)]}
        dictionary of files assigned to speaker id
    n_splits : int
        number of groups, that traing set will be divided to
    cmv_norm : bool, deafult False
        cepstral mean and variance normalization
    kwargs : additional keyword arguments
        Arguments to `librosa.feature.mfcc`

    Returns
    -------
    deltas : bool
    """
    speaker_mfccs = data_processing.generate_speaker_mfccs(speaker_files, cmv_norm=cmv_norm, deltas=False, **kwargs)
    speaker_mfccs_deltas = data_processing.generate_speaker_mfccs(speaker_files, cmv_norm=cmv_norm, deltas=True, **kwargs)

    default_gmm_args = {'n_components': 1, 'covariance_type': 'diag'}

    acc_without_deltas = cross_validate(speaker_mfccs, n_splits, **default_gmm_args)[0]
    acc_with_deltas = cross_validate(speaker_mfccs_deltas, n_splits, **default_gmm_args)[0]

    print("Classification accuracy without deltas of MFCCs: {:.3f}".format(acc_without_deltas))
    print("Classification accuracy with deltas of MFCCs: {:.3f}".format(acc_with_deltas))

    if acc_without_deltas >= acc_with_deltas:
        deltas = False
    else:
        deltas = True
    
    return deltas



def test_mfcc_params(speaker_files, n_splits, parameter_name, value_range, cmv_norm=False, **kwargs):
    """Tests model accuracy according to given parameter

    This function generates plot with results.

    Parameters
    ----------
    speaker_files : dict of {str : list of np.ndarray [shape=(n,)]}
        dictionary of files assigned to speaker id
    n_splits : int
        number of groups, that traing set will be divided to
    parameter_name : str
        parameter name in librosa documentation
    value_range : list of int or float, range object
        parameter values to test
    cmv_norm : bool, deafult False
        cepstral mean and variance normalization
    kwargs : additional keyword arguments
        Arguments to `librosa.feature.mfcc`

    Returns
    -------
    best_param_value : int or float
    """
    default_gmm_args = {'n_components': 1, 'covariance_type': 'diag'}

    accuracy = []
    accuracy_err = []

    beast_mean = -1
    for i in value_range:
        kwargs[parameter_name] = i
        speaker_mfccs = data_processing.generate_speaker_mfccs(speaker_files, cmv_norm=cmv_norm, deltas=False, **kwargs)
        mean_acc, stdev_acc = cross_validate(speaker_mfccs, n_splits, **default_gmm_args)
        accuracy.append(mean_acc)
        accuracy_err.append(stdev_acc)
        if mean_acc > beast_mean:
            beast_mean = mean_acc
            best_param_value = i
    
    fig, ax = plt.subplots(figsize=(9,5))
    ax.errorbar(value_range, accuracy, yerr=accuracy_err)
    ax.set_xlabel('{} value'.format(parameter_name))
    ax.set_ylabel('Accuracy')
    ax.set_title('MFCCs {} parameter'.format(parameter_name))
    ax.set_xticks(value_range)
    ax.set_ylim((0, 1))
    ax.grid(True)
    fig.tight_layout()
    plt.show()

    return best_param_value



def evaluate(gmms, data_path, output_filename, cmv_norm=False, deltas=False, **kwargs):
    """Generates .csv file with predictions and likelihoods

    Format of .csv file is: filename, predicted digit, loglikelihood, 
    so for example: 0001.wav,1,-56.37

    Parameters
    ----------
    gmms : list of `sklearn.mixture.GaussianMixture` objects
        list of GMM models
    data_path : str
        path to evaluation files
    output_filename : str
        name of output .csv file
    cmv_norm : bool, deafult False
        cepstral mean and variance normalization
    deltas : bool, deafult False
        compute local estimate of the derivative of the mfcc data
    kwargs : additional keyword arguments
        Arguments to `librosa.feature.mfcc`
    """
    wav_filenames = [filename for filename in listdir(data_path) if isfile(join(data_path, filename)) and filename.endswith('.wav')]
    wav_filenames.sort()

    with open(output_filename, 'w',  newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        for filename in wav_filenames:
            predicted_digit, log_likelihood = model_train.predict_digit(gmms, file_path=join(data_path, filename), cmv_norm=cmv_norm, deltas=deltas, **kwargs)
            writer.writerow([filename, predicted_digit, log_likelihood])