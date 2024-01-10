
# Libraries 

import os
import math
import random
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import signal
from scipy.io import loadmat
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.utils import resample
from sklearn.preprocessing import LabelEncoder
from imblearn.datasets import make_imbalance
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.feature_selection import RFE, RFECV
from imblearn.over_sampling import SMOTE

import tensorflow as tf
from tensorflow import keras
from keras.utils import plot_model
from keras.utils import pad_sequences

from tqdm import tqdm
from biosppy.signals.tools import filter_signal


def onehot_encode(df_labels):
    one_hot = MultiLabelBinarizer()
    y=one_hot.fit_transform(df_labels[0].str.split(pat=','))
    print("The classes we will look at are encoded as SNOMED CT codes:")
    print(one_hot.classes_)
    y = np.delete(y, -1, axis=1)
    print("classes: {}".format(y.shape[1]))
    return y, one_hot.classes_[0:-1]


def load_challenge_data(filename):
    x = loadmat(filename)
    data = np.asarray(x['val'], dtype=np.float64)
    new_file = filename.replace('.mat','.hea')
    input_header_file = os.path.join(new_file)
    with open(input_header_file,'r') as f:
        header_data=f.readlines()
    fs = int(header_data[0].split()[2])
    if fs!=500:
        print('yes', i)
        data = signal.resample(data, fs)
    
    return data,fs


def make_undefined_class(labels, df_unscored):
    df_labels = pd.DataFrame(labels)
    labs = df_unscored.iloc[:,1].to_list()
    extra = [67751000119106,55827005,251199005,251198002,251166008,233897008,
            17366009,164942001,164912004]
    conc = labs + extra
    
    for i in range(len(conc)):
        df_labels.replace(to_replace=str(conc[i]), inplace=True ,value="undefined class", regex=True)

    return df_labels


def get_labels_for_all_combinations(y):
    y_all_combinations = LabelEncoder().fit_transform([''.join(str(l)) for l in y])
    return y_all_combinations


def split_data(labels, y_all_combo):
    folds = list(StratifiedKFold(n_splits=4, shuffle=True, random_state=20).split(labels,y_all_combo))
    print("Training split: {}".format(len(folds[0][0])))
    print("Validation split: {}".format(len(folds[0][1])))
    print("All data: {}".format(len(folds[0][1])+len(folds[0][0])))
    return folds
        

def shuffle_batch_generator(batch_size, gen_x,gen_y,snomed_classes): 
    batch_features = np.zeros((batch_size,5000, 12))
    batch_labels = np.zeros((batch_size,snomed_classes.shape[0])) #drop undef class
    while True:
        for i in range(batch_size):

            batch_features[i] = next(gen_x)
            batch_labels[i] = next(gen_y)
            
        yield batch_features, batch_labels

def generate_y_shuffle(y_train):
    while True:
        for i in range(len(y_train)):
            y_shuffled = y_train[i]
            yield y_shuffled


def generate_X_shuffle(X_train):
    while True:
        for i in range(len(X_train)):
                    data,_ = load_challenge_data(X_train[i])
                    X_train_new = pad_sequences(data, maxlen=5000, truncating='post',padding="post")
                    X_train_new = X_train_new.reshape(5000,12)
                    X_train_new = X_train_new / abs(X_train_new).max()

                    yield X_train_new


def calculating_class_weights(y_true):
    number_dim = np.shape(y_true)[1]
    weights = np.empty([number_dim, 2])
    for i in range(number_dim):
        weights[i] = compute_class_weight('balanced',classes=[0.,1.], y= y_true[:, i])
    return weights


def generate_validation_data(ecg_filenames, y):
    y_train_gridsearch=y
    ecg_filenames_train_gridsearch=ecg_filenames
    ecg_train_timeseries=[]
    
    for names in ecg_filenames_train_gridsearch:
        
        data,_ = load_challenge_data(names)
        data = pad_sequences(data, maxlen=5000, truncating='post',padding="post")
        data = data / abs(data).max()
        ecg_train_timeseries.append(data)
    X_train_gridsearch = np.asarray(ecg_train_timeseries)

    X_train_gridsearch = X_train_gridsearch.reshape(ecg_filenames_train_gridsearch.shape[0],5000,12)

    return X_train_gridsearch, y_train_gridsearch


def residual_network_1d(Dense_num,trainable,trainable_last_layer=True,trainableOnelast=True,Classifire=1,LR=10e-3):
    n_feature_maps = 64
    input_shape = (None,12)
    input_layer = keras.layers.Input(input_shape)

    # BLOCK 1

    conv_x = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=12, padding='same',trainable=trainable)(input_layer)
    conv_x = keras.layers.BatchNormalization(trainable=trainable)(conv_x)
    conv_x = keras.layers.Activation('relu',trainable=trainable)(conv_x)

    conv_y = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=8, padding='same',trainable=trainable)(conv_x)
    conv_y = keras.layers.BatchNormalization(trainable=trainable)(conv_y)
    conv_y = keras.layers.Activation('relu',trainable=trainable)(conv_y)

    conv_z = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=5, padding='same',trainable=trainable)(conv_y)
    conv_z = keras.layers.BatchNormalization(trainable=trainable)(conv_z)

    # expand channels for the sum
    shortcut_y = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=1, padding='same',trainable=trainable)(input_layer)
    shortcut_y = keras.layers.BatchNormalization(trainable=trainable)(shortcut_y)

    output_block_1 = keras.layers.add([shortcut_y, conv_z],trainable=trainable)
    output_block_1 = keras.layers.Activation('relu',trainable=trainable)(output_block_1)

    # BLOCK 2

    conv_x = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=12, padding='same',trainable=trainable)(output_block_1)
    conv_x = keras.layers.BatchNormalization(trainable=trainable)(conv_x)
    conv_x = keras.layers.Activation('relu',trainable=trainable)(conv_x)

    conv_y = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=8, padding='same',trainable=trainable)(conv_x)
    conv_y = keras.layers.BatchNormalization(trainable=trainable)(conv_y)
    conv_y = keras.layers.Activation('relu',trainable=trainable)(conv_y)

    conv_z = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=5, padding='same',trainable=trainable)(conv_y)
    conv_z = keras.layers.BatchNormalization(trainable=trainable)(conv_z)

    # expand channels for the sum
    shortcut_y = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=1, padding='same',trainable=trainable)(output_block_1)
    shortcut_y = keras.layers.BatchNormalization(trainable=trainable)(shortcut_y)

    output_block_2 = keras.layers.add([shortcut_y, conv_z],trainable=trainable)
    output_block_2 = keras.layers.Activation('relu',trainable=trainable)(output_block_2)

    # BLOCK 3

    conv_x = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=12, padding='same',trainable=trainableOnelast)(output_block_2)
    conv_x = keras.layers.BatchNormalization(trainable=trainableOnelast)(conv_x)
    conv_x = keras.layers.Activation('relu',trainable=trainableOnelast)(conv_x)

    conv_y = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=8, padding='same',trainable=trainableOnelast)(conv_x)
    conv_y = keras.layers.BatchNormalization(trainable=trainableOnelast)(conv_y)
    conv_y = keras.layers.Activation('relu',trainable=trainableOnelast)(conv_y)

    conv_z = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=5, padding='same',trainable=trainableOnelast)(conv_y)
    conv_z = keras.layers.BatchNormalization(trainable=trainableOnelast)(conv_z)

    # no need to expand channels because they are equal
    shortcut_y = keras.layers.BatchNormalization(trainable=trainableOnelast)(output_block_2)

    output_block_3 = keras.layers.add([shortcut_y, conv_z],trainable=trainableOnelast)
    output_block_3 = keras.layers.Activation('relu',trainable=trainableOnelast)(output_block_3)
    
    # FINAL

    gap_layer = keras.layers.GlobalAveragePooling1D(trainable=trainableOnelast)(output_block_3)
    
    if Classifire == 2:
    
        output_layer = keras.layers.Dense(Dense_num*6, activation='relu',trainable=trainable_last_layer)(gap_layer)

        output_layer = keras.layers.Dense(Dense_num, activation='softmax',trainable=trainable_last_layer)(output_layer)

    else:
        output_layer = keras.layers.Dense(Dense_num, activation='softmax',trainable=trainable_last_layer)(gap_layer)
   
    model = keras.models.Model(inputs=input_layer, outputs=output_layer)
    
    
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer=tf.keras.optimizers.Adam(learning_rate=LR), metrics=[tf.keras.metrics.BinaryAccuracy(
    name='accuracy', dtype=None, threshold=0.5),tf.keras.metrics.Recall(name='Recall'),tf.keras.metrics.Precision(name='Precision'), 
                tf.keras.metrics.AUC(
    num_thresholds=200,
    curve="ROC",
    summation_method="interpolation",
    name="AUC",
    dtype=None,
    thresholds=None,
    multi_label=True,
    label_weights=None,
    )])

    return model