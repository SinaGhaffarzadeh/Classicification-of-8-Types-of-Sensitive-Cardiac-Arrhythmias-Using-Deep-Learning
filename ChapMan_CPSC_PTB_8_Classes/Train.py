
"""

Final part of training

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
NumOfClass = 8
NumOfEpochs = 100
train_split = 0
test_split = 1
batchsize = 24

Chap_CPSC_PTB = [] 

All_8Class = []

# -------------------------------------------Preprocessing part-------------------------------------------------------

# Make a Data Frame with direction for K-fold of ChapMan, CPSC, and PTB dataset using Dirction_and_labels file


# Make train and test data frame 
for fold in range(NumOfFold):
    
    Chap_CPSC_PTB_ = []
    
    for TrTe in range(2):
        
        Chap_CPSC_PTB_df = Dirction_and_labels.iloc[Folds_splited_data[fold][TrTe]]
        Chap_CPSC_PTB_df = Chap_CPSC_PTB_df.sample(frac = 1,random_state=42)
        Chap_CPSC_PTB_.append(Chap_CPSC_PTB_df.set_index(pd.Index(np.arange(0,len(Chap_CPSC_PTB_df)))))
    
    Chap_CPSC_PTB.append([Chap_CPSC_PTB_[0],Chap_CPSC_PTB_[1]])


# Binarizing our k-fold labels
for fold in range(NumOfFold):
    
    for TrTe in range(2): 

        one_hot = MultiLabelBinarizer()
        y_=one_hot.fit_transform(Chap_CPSC_PTB[fold][TrTe].CT_code.str.split(pat=','))
        print("The classes we will look at are encoded as SNOMED CT codes:")
        print(one_hot.classes_)
        y_1 = np.delete(y_, -1, axis=1)
        print("classes: {}".format(y_1.shape[1]))

        Chap_CPSC_PTB[fold][TrTe]['Labs'] = list(y_1)
        snomed_classes = one_hot.classes_[0:-1]



# Reduce number of All 8 rhythms with a constant coefficient to number of PAC rhythm 
for fold in range(NumOfFold):
    All_8Class_infold = []
    
    for TrTe in range(2):
        
        PAC_Rhythm = Chap_CPSC_PTB[fold][TrTe].Labs_Type[Chap_CPSC_PTB[fold][TrTe].Labs_Type=='PAC'].index
        PVC_Rhythm = Chap_CPSC_PTB[fold][TrTe].Labs_Type[Chap_CPSC_PTB[fold][TrTe].Labs_Type=='PVC'].index
        SIN_Rhythm = Chap_CPSC_PTB[fold][TrTe].Labs_Type[Chap_CPSC_PTB[fold][TrTe].Labs_Type=='SIN'].index
        SVT_Rhythm = Chap_CPSC_PTB[fold][TrTe].Labs_Type[Chap_CPSC_PTB[fold][TrTe].Labs_Type=='SVT'].index
        SB_Rhythm = Chap_CPSC_PTB[fold][TrTe].Labs_Type[Chap_CPSC_PTB[fold][TrTe].Labs_Type=='SB'].index
        STach_Rhythm = Chap_CPSC_PTB[fold][TrTe].Labs_Type[Chap_CPSC_PTB[fold][TrTe].Labs_Type=='STach'].index
        Afib_Rhythm = Chap_CPSC_PTB[fold][TrTe].Labs_Type[Chap_CPSC_PTB[fold][TrTe].Labs_Type=='Afib'].index
        AF_Rhythm = Chap_CPSC_PTB[fold][TrTe].Labs_Type[Chap_CPSC_PTB[fold][TrTe].Labs_Type=='Af'].index

        if TrTe == 0:
            fraction = 5
        else:
            fraction = 5
            
        PVC_Rhythm_Sampled = Chap_CPSC_PTB[fold][TrTe].iloc[PVC_Rhythm]
        PVC_Rhythm_Sampled = PVC_Rhythm_Sampled.sample(n=len(PVC_Rhythm), random_state=42,replace=True)

        PAC_Rhythm_Sampled = Chap_CPSC_PTB[fold][TrTe].iloc[PAC_Rhythm]
        PAC_Rhythm_Sampled = PAC_Rhythm_Sampled.sample(n=len(PVC_Rhythm), random_state=42)

        SIN_Rhythm_Sampled = Chap_CPSC_PTB[fold][TrTe].iloc[SIN_Rhythm]
        SIN_Rhythm_Sampled = SIN_Rhythm_Sampled.sample(n=len(PVC_Rhythm)-fraction, random_state=42)

        SVT_Rhythm_Sampled = Chap_CPSC_PTB[fold][TrTe].iloc[SVT_Rhythm]
        SVT_Rhythm_Sampled = SVT_Rhythm_Sampled.sample(n=len(PVC_Rhythm)-fraction, random_state=42,replace=True)

        SB_Rhythm_Sampled = Chap_CPSC_PTB[fold][TrTe].iloc[SB_Rhythm]
        SB_Rhythm_Sampled = SB_Rhythm_Sampled.sample(n=len(PVC_Rhythm)-fraction, random_state=42)

        STach_Rhythm_Sampled = Chap_CPSC_PTB[fold][TrTe].iloc[STach_Rhythm]
        STach_Rhythm_Sampled = STach_Rhythm_Sampled.sample(n=len(PVC_Rhythm)-fraction, random_state=42)

        Afib_Rhythm_Sampled = Chap_CPSC_PTB[fold][TrTe].iloc[Afib_Rhythm]
        Afib_Rhythm_Sampled = Afib_Rhythm_Sampled.sample(n=len(PVC_Rhythm)-fraction, random_state=42)

        AF_Rhythm_Sampled = Chap_CPSC_PTB[fold][TrTe].iloc[AF_Rhythm]
        AF_Rhythm_Sampled = AF_Rhythm_Sampled.sample(n=len(PVC_Rhythm)-fraction, random_state=42,replace=True)
        
        Chap_CPSC_PTB_df = pd.concat([SIN_Rhythm_Sampled,SVT_Rhythm_Sampled,SB_Rhythm_Sampled,STach_Rhythm_Sampled,
                                                      Afib_Rhythm_Sampled,AF_Rhythm_Sampled,PAC_Rhythm_Sampled,PVC_Rhythm_Sampled],ignore_index=True)
        Chap_CPSC_PTB_df = Chap_CPSC_PTB_df.sample(frac = 1,random_state=42)
        Chap_CPSC_PTB_df = Chap_CPSC_PTB_df.set_index(pd.Index(np.arange(0,len(Chap_CPSC_PTB_df))))

        All_8Class_infold.append(Chap_CPSC_PTB_df)


    All_8Class.append(All_8Class_infold)



# ----------------------------------------------Training part--------------------------------------------------------

    """we used learning rate reducer, early stopping, and Checkpoint function in training process"""

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau( monitor='val_Recall', factor=0.1, patience=1, verbose=1, mode='max',
    min_delta=0.0001, cooldown=0, min_lr=0)
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_Recall', mode='max', verbose=1, patience=20)

for fold in range(NumOfFold):

    checkpoint_filepath =f'.../Weighs/Model_weights_{fold}_8Class_Chap_CPSC_PTB.hdf5'
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_Recall',
        mode='max',
        save_best_only=True)
    
    last_model = functions.residual_network_1d(NumOfClass-1,trainable=True,trainable_last_layer=True,trainableOnelast=True,Classifire=1,LR=10e-3)
    model = functions.residual_network_1d(NumOfClass,trainable=False,trainable_last_layer=True,trainableOnelast=True,Classifire=2,LR=1e-3)
    last_model.load_weights(f'...\ChapMan_CPSC_PTB_7_Classes\Weights\Model_weights_{fold}_7Class_Chap_CPSC_PTB_with_PAC.hdf5')
    
    for i in range(len(model.layers)-3):
        model.layers[i].set_weights(last_model.layers[i].get_weights())
    
    # order_array = folds[foldd][0]
    history = model.fit(x=functions.shuffle_batch_generator(batch_size=batchsize,
                                        gen_x=functions.generate_X_shuffle(np.asarray(All_8Class[fold][0].Ecg_dir.tolist())),
                                        gen_y=functions.generate_y_shuffle(np.asarray(All_8Class[fold][0].Labs.tolist())),snomed_classes=snomed_classes),
              epochs=NumOfEpochs, steps_per_epoch=(len(np.asarray(All_8Class[fold][0].Ecg_dir.tolist()))/batchsize),
              validation_data=functions.generate_validation_data(np.asarray(All_8Class[fold][1].Ecg_dir.tolist()),np.asarray(All_8Class[fold][1].Labs.tolist())), 
              validation_freq=1,callbacks=[reduce_lr,early_stop,model_checkpoint_callback])