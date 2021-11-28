# imports

import numpy as np
import pandas as pd
import seaborn as sbn
import collections
import os
import datetime
import shap
import math
from numpy import dot
from numpy.linalg import norm
from scipy.stats import entropy
from sklearn.metrics import classification_report
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.get_logger().setLevel('INFO')
tf.autograph.set_verbosity(1)
tf.compat.v1.disable_v2_behavior()
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding
from tensorflow.keras.layers import LSTM
from tensorflow.keras.datasets import imdb

from tqdm import tqdm

#Global Variables
RANDOM_SEED = 42
max_features = 20000
maxlen = 80                                                                     # cut texts after this number of words (among top max_features most common words)
batch_size = 32
EPOCHS = 3
q, c, p, b, k, initial_seed_size, initial_seed, j, S_indices, U_indices, train_indices, model_theta = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0

def load_data():
    print('Loading data...')
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features, seed=RANDOM_SEED)
    print(len(x_train), 'train sequences')
    print(len(x_test), 'test sequences')

    print('Pad sequences (samples x time)')
    x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
    x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
    print('x_train shape:', x_train.shape)
    print('x_test shape:', x_test.shape)

    return x_train, y_train, x_test, y_test

def get_lstm_model():
    print('Build model...')
    model_lstm = Sequential()
    model_lstm.add(Embedding(max_features, 128))
    model_lstm.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
    model_lstm.add(Dense(2, activation='softmax'))

    model_lstm.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model_lstm

def init_workflow():
    global q, c, p, b, k, initial_seed_size, initial_seed, j, S_indices, U_indices, model_theta, train_indices
    np.random.seed(RANDOM_SEED)
    p = 10
    train_indices = [i for i in range(len(x_train))]
    initial_seed_size = int(len(x_train) * 0.005)
    b = initial_seed_size
    k = 2 * initial_seed_size
    initial_seed = np.random.choice(train_indices, size=initial_seed_size, replace=False)
    j = 1
    S_indices = initial_seed
    U_indices = list(set(train_indices) - set(S_indices))

def train_model_lstm(model_lstm, x_train, y_train):
    print('Train...')
    model_lstm.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=EPOCHS,
              validation_data=(x_test, y_test))
    score, acc = model_lstm.evaluate(x_test, y_test,
                                batch_size=batch_size)
    print('Test score:', score)
    print('Test accuracy:', acc)

    return model_lstm


def compute_cosine_similarity(term_dict_1, term_dict_2):
    cosine_sim = 0
    for index in term_dict_1.keys():
        if(index in term_dict_2):
            cosine_sim += term_dict_1[index] * term_dict_2[index]
    values_term_1 = sum([v**2 for k, v in term_dict_1.items()])
    values_term_2 = sum([v**2 for k, v in term_dict_2.items()])

    cosine_sim /= ( math.sqrt(values_term_1) * math.sqrt(values_term_2) )

    return cosine_sim

def alex_sampling_method(x_train, y_train, x_test, y_test):

    global q, c, p, b, k, initial_seed_size, initial_seed, j, S_indices, U_indices, model_theta, train_indices
    init_workflow()

    while(j <= p):
        print(f'==========Active Learning Iteration {j}/{p}==========')
        x_train_subset = x_train[S_indices]
        y_train_subset = y_train[S_indices]

        print('Training Prediction Model')
        model_theta = get_lstm_model()
        model_theta = train_model_lstm(model_theta, x_train_subset, y_train_subset)

        train_preds = np.argmax(model_theta.predict(x_train_subset), axis=1)

        print('Generating Classification Report')
        y_pred = np.argmax(model_theta.predict(x_test), axis=1)
        print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))

        if(j == p):
            break

        print('Training Explainer Model on Labelled Set')
        explainer = shap.DeepExplainer(model_theta, x_train_subset)

        print(f'Computing Shap values for labelled instances - {len(x_train_subset)}')
        shap_values_labelled = explainer.shap_values(x_train_subset)

        print('Predicting unlabelled instances')
        s_x = np.ndarray((len(U_indices), 3))
        preds = model_theta.predict(x_train[U_indices])
        # s_x[:, 0] = np.amax(preds, axis=1)
        s_x[:, 0] = entropy(np.array(preds), base=2,axis=1)
        s_x[:, 1] = U_indices
        s_x[:, 2] = np.argmax(preds, axis=1)

        sort_index = np.argsort(-s_x[:, 0])
        s_x = s_x[sort_index]
        c_indxs = list(map(int, s_x[:k, 1]))

        pred_y_dict = dict(zip(s_x[:k, 1], s_x[:k, 2]))

        print('Training Explainer Model on Unlabelled Candidate Set')
        explainer = shap.DeepExplainer(model_theta, x_train[c_indxs])

        print(f'Computing Shap values for unlabelled candidate set instances - {len(x_train[c_indxs])}')
        shap_values_unlabelled = explainer.shap_values(x_train[c_indxs])

        print('Computing Similarities')
        xu_kld = np.ndarray((len(c_indxs), 2))
        for i, index_xu in enumerate(tqdm(c_indxs)):
            term_dict_U = dict(zip(x_train[int(index_xu)], shap_values_unlabelled[int(pred_y_dict[int(index_xu)])][i]))
            for q, index_x in enumerate(S_indices):
                term_dict_S = dict(zip(x_train[int(index_x)], shap_values_labelled[y_train[int(index_x)]][q]))
                xu_kld[i][0] += compute_cosine_similarity(term_dict_S, term_dict_U)
            xu_kld[i][0] /= len(S_indices)
            xu_kld[i][1] = index_xu

        sort_index = np.argsort(xu_kld[:, 0])
        xu_kld = xu_kld[sort_index]                                             #least similar first; get such top k samples
        delta_S = xu_kld[:b, 1]
        print(f'Adding {len(delta_S)} samples to labelled set.')
        S_indices = np.append(S_indices, list(map(int, delta_S)))
        print(f'Total samples in labelled set {len(S_indices)}')
        print(f'Removing {len(delta_S)} samples from unlabelled set')
        U_indices = list(set(train_indices) - set(S_indices))
        print(f'Total samples in unlabelled set {len(U_indices)}')
        j += 1

def random_sampling_method(x_train, y_train, x_test, y_test):
    global q, c, p, b, k, initial_seed_size, initial_seed, j, S_indices, U_indices, model_theta, train_indices
    init_workflow()

    while(j <= p):
        print(f'==========Active Learning Iteration {j}/{p}==========')
        x_train_subset = x_train[S_indices]
        y_train_subset = y_train[S_indices]

        print('Training Prediction Model')
        model_theta = get_lstm_model()
        model_theta = train_model_lstm(model_theta, x_train_subset, y_train_subset)

        if(j == p):
            break

        print(f'Selecting random {initial_seed_size} samples')
        delta_S = np.random.choice(U_indices, size=initial_seed_size, replace=False)

        print(f'Adding {len(delta_S)} samples to labelled set.')
        S_indices = np.append(S_indices, list(map(int, delta_S)))
        print(f'Total samples in labelled set {len(S_indices)}')

        print(f'Removing {len(delta_S)} samples from unlabelled set')
        U_indices = list(set(train_indices) - set(S_indices))
        print(f'Total samples in unlabelled set {len(U_indices)}')

        U_indices = list(set(train_indices) - set(S_indices))
        j += 1

def uncertainty_sampling_least_confidence_method(x_train, y_train, x_test, y_test):
    global q, c, p, b, k, initial_seed_size, initial_seed, j, S_indices, U_indices, model_theta, train_indices
    init_workflow()

    while(j <= p):
        print(f'==========Active Learning Iteration {j}/{p}==========')
        x_train_subset = x_train[S_indices]
        y_train_subset = y_train[S_indices]

        print('Training Prediction Model')
        model_theta = get_lstm_model()
        model_theta = train_model_lstm(model_theta, x_train_subset, y_train_subset)

        if(j == p):
            break

        print('Predicting unlabelled instances and computing uncertainty by least confidence')
        s_x = np.ndarray((len(U_indices), 2))
        s_x[:, 0] = 1-np.amax(model_theta.predict(x_train[U_indices]), axis=1)
        s_x[:, 1] = U_indices
        sort_index = np.argsort(s_x[:, 0])
        s_x = s_x[-sort_index]   #larger the (1-model_top_pred) difference, lower the confidence of the model in that prediction, get such top k samples
        delta_S = s_x[:k, 1]

        print(f'Adding {len(delta_S)} samples to labelled set.')
        S_indices = np.append(S_indices, list(map(int, delta_S)))
        print(f'Total samples in labelled set {len(S_indices)}')

        print(f'Removing {len(delta_S)} samples from unlabelled set')
        U_indices = list(set(train_indices) - set(S_indices))
        print(f'Total samples in unlabelled set {len(U_indices)}')

        j += 1

def uncertainty_sampling_smallest_margin_method(x_train, y_train, x_test, y_test):
    global q, c, p, b, k, initial_seed_size, initial_seed, j, S_indices, U_indices, model_theta, train_indices
    init_workflow()

    while(j <= p):
        print(f'==========Active Learning Iteration {j}/{p}==========')
        x_train_subset = x_train[S_indices]
        y_train_subset = y_train[S_indices]

        print('Training Prediction Model')
        model_theta = get_lstm_model()
        model_theta = train_model_lstm(model_theta, x_train_subset, y_train_subset)

        if(j == p):
            break

        print('Predicting unlabelled instances and computing uncertainty by smallest margin')
        s_x = np.ndarray((len(U_indices), 2))
        pred = model_theta.predict(x_train[U_indices])
        s_x[:, 0] = abs(pred[:, 0] - pred[:, 1])
        s_x[:, 1] = U_indices
        sort_index = np.argsort(s_x[:, 0])
        s_x = s_x[sort_index]   #smallest difference (margin) means model is struggling to differentiate between the two classes; take such top k samples
        delta_S = s_x[:k, 1]

        print(f'Adding {len(delta_S)} samples to labelled set.')
        S_indices = np.append(S_indices, list(map(int, delta_S)))
        print(f'Total samples in labelled set {len(S_indices)}')

        print(f'Removing {len(delta_S)} samples from unlabelled set')
        U_indices = list(set(train_indices) - set(S_indices))
        print(f'Total samples in unlabelled set {len(U_indices)}')

        j += 1
if __name__ == '__main__':

    total_time_start = datetime.datetime.now().replace(microsecond=0)

    args = sys.argv[1:]
    if("-A" in args):
        args.remove('-A')
        args.append('-a')
        args.append('-r')
        args.append('-l')
        args.append('-m')

    if(not args):
        print()
        print(f'****Arguments Not Found****')
        print('USAGE: python AL_TextClassification.py [FLAGS]')
        print('Available Flags:\n-a : ALEX Method\n-r : Random Sampling Method\n-l : Uncertainty Sampling Least Confidence Method\n-m : Uncertainty Sampling Smallest Margin Method\n-A : Run all methods')
        print('Multiple Flags space separated: python AL_TextClassification.py -a -r')
    else:
        for arg in args:
            try:
                if(arg == "-a"):
                    print()
                    print('Running ALEX Method')
                    x_train, y_train, x_test, y_test = load_data()
                    alex_sampling_method(x_train, y_train, x_test, y_test)
                elif(arg == "-r"):
                    print()
                    print('Running Random Sampling Method')
                    x_train, y_train, x_test, y_test = load_data()
                    random_sampling_method(x_train, y_train, x_test, y_test)
                elif(arg == "-l"):
                    print()
                    print('Running Uncertainty Sampling Least Confidence Method')
                    x_train, y_train, x_test, y_test = load_data()
                    uncertainty_sampling_least_confidence_method(x_train, y_train, x_test, y_test)
                elif(arg == "-m"):
                    print()
                    print('Running Uncertainty Sampling Smallest Margin Method')
                    x_train, y_train, x_test, y_test = load_data()
                    uncertainty_sampling_smallest_margin_method(x_train, y_train, x_test, y_test)
                else:
                    print()
                    print(f'****Invalid Flag {arg}****')
                    print('USAGE: python AL_TextClassification.py [FLAGS]')
                    print('Available Flags:\n-a : ALEX Method\n-r : Random Sampling Method\n-l : Uncertainty Sampling Least Confidence Method\n-m : Uncertainty Sampling Smallest Margin Method\n-A : Run all methods')
                    print('Multiple Flags space separated: python AL_TextClassification.py -a -r')
            except Exception as e:
                print()
                print('****Exception Occured****')
                print(e)

    total_time_end = datetime.datetime.now().replace(microsecond=0)
    print(f'Total time: {total_time_end-total_time_start}')
