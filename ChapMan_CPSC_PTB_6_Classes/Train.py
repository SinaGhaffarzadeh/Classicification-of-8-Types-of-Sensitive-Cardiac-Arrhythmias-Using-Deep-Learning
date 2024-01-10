
"""

Second part of training

"""

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

from biosppy.signals.tools import filter_signal

from functions import functions

# ------------------------------------------Reading existing files---------------------------------------------------- 

Dirction_and_labels = pd.read_excel('...\Direction_And_folds\Chap_CPSC_PTB_Direction_SingleLabels_CT-Code.xlsx')

with open('...\Direction_And_folds\Train_Test_Split_8Class_Chap_CPSC_PTB.pickle', 'rb') as handle:
    Folds_splited_data = pickle.load(handle)


# ------------------------------------------Constant values and Empty lists-------------------------------------------

NumOfFold = 6
NumOfClass = 6
NumOfEpochs = 100
train_split = 0
test_split = 1
batchsize = 24


train_split = 0
test_split = 1
PAC_Rhythm = []
PVC_Rhythm = []
SIN_Rhythm = []
Chap_CPSC_PTB_df_Without_PAC_PVC = [] 

# -------------------------------------------Preprocessing part-------------------------------------------------------

# Make a Data Frame with direction for K-fold of ChapMan, CPSC, and PTB dataset using Dirction_and_labels file

# Extract the index of PAC and PVC rhythms to remove them from our Data Frame
for fold in range(NumOfFold):
    PAC = []
    PVC = []
    SIN = []
    for TrTe in range(2):

        PAC.append(Dirction_and_labels.iloc[Folds_splited_data[fold][TrTe]].Labs_Type[Dirction_and_labels.iloc[Folds_splited_data[fold][TrTe]].Labs_Type=='PAC'].index)
        PVC.append(Dirction_and_labels.iloc[Folds_splited_data[fold][TrTe]].Labs_Type[Dirction_and_labels.iloc[Folds_splited_data[fold][TrTe]].Labs_Type=='PVC'].index)
        SIN.append(Dirction_and_labels.iloc[Folds_splited_data[fold][TrTe]].Labs_Type[Dirction_and_labels.iloc[Folds_splited_data[fold][TrTe]].Labs_Type=='SIN'].index)

    PAC_Rhythm.append([PAC[0],PAC[1]])
    PVC_Rhythm.append([PVC[0],PVC[1]])
    SIN_Rhythm.append([SIN[0],SIN[1]])


# Make train and test data frame without PVC and PAC and a fraction of Normal
for fold in range(NumOfFold):
    Chap_CPSC_PTB = []

    for TrTe in range(2):

        NO_samp = np.asarray(list(Dirction_and_labels.iloc[Folds_splited_data[fold][TrTe]]['Labs'])).sum(axis=0)
        NO_samp.sort()

        if TrTe == 0:
            thre = 500
        else:
            thre = 200
        NO_New_SIN = NO_samp[-1] - (NO_samp[-2]+thre)

        Chap_CPSC_PTB_df = Dirction_and_labels.iloc[Folds_splited_data[fold][TrTe]].drop(PAC_Rhythm[fold][TrTe].tolist() + PVC_Rhythm[fold][TrTe].tolist() + random.sample(SIN_Rhythm[fold][TrTe].tolist(), NO_New_SIN))
        Chap_CPSC_PTB.append(Chap_CPSC_PTB_df.set_index(pd.Index(np.arange(0,len(Chap_CPSC_PTB_df)))).sample(frac = 1,random_state=42))

    ChapMan_CPSC_PTB_Without_PAC_PVC.append([Chap_CPSC_PTB[0],Chap_CPSC_PTB[1]])


# Binarizing our k-fold labels
for fold in range(NumOfFold):

    for TrTe in range(2):

        one_hot = MultiLabelBinarizer()
        y_=one_hot.fit_transform(ChapMan_CPSC_PTB_Without_PAC_PVC[fold][TrTe].CT_code.str.split(pat=','))
        print("The classes we will look at are encoded as SNOMED CT codes:")
        print(one_hot.classes_)
        y_1 = np.delete(y_, -1, axis=1)
        print("classes: {}".format(y_1.shape[1]))

        ChapMan_CPSC_PTB_Without_PAC_PVC[fold][TrTe]['Labs'] = list(y_1)
        snomed_classes = one_hot.classes_[0:-1]

# ----------------------------------------------Training part--------------------------------------------------------

    """we used learning rate reducer, early stopping, and Checkpoint function in training process"""

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau( monitor='val_Recall', factor=0.1, patience=1, verbose=1, mode='max',
    min_delta=0.0001, cooldown=0, min_lr=0)

early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_Recall', mode='max', verbose=1, patience=5)

for fold in range(NumOfFold):

    checkpoint_filepath =f'.../Model_weights_{fold}_6Class_Chap_CPSC_PTB.hdf5'
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_Recall',
        mode='max',
        save_best_only=True)

    model = functions.residual_network_1d(NumOfClass,trainable=True,trainable_last_layer=True,trainableOnelast=True,Classifire=1,LR=10e-3)
    model.load_weights(f'.../res_fold_{fold}_6Class_ChapMan.hdf5')

    history = model.fit(x=functions.shuffle_batch_generator(batch_size=batchsize,
                                        gen_x=functions.generate_X_shuffle(np.asarray(ChapMan_CPSC_PTB_Without_PAC_PVC[fold][0].Ecg_dir.tolist())),
                                        gen_y=functions.generate_y_shuffle(np.asarray(ChapMan_CPSC_PTB_Without_PAC_PVC[fold][0].Labs.tolist())),snomed_classes=snomed_classes),
              epochs=NumOfEpochs, steps_per_epoch=(len(np.asarray(ChapMan_CPSC_PTB_Without_PAC_PVC[fold][0].Ecg_dir.tolist()))/batchsize),
              validation_data=functions.generate_validation_data(np.asarray(ChapMan_CPSC_PTB_Without_PAC_PVC[fold][1].Ecg_dir.tolist()),np.asarray(ChapMan_CPSC_PTB_Without_PAC_PVC[fold][1].Labs.tolist())),
              validation_freq=1, callbacks=[reduce_lr,early_stop,model_checkpoint_callback])
