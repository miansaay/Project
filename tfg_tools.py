#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 19:21:23 2017

@author: obarquero
"""
import scipy.io as io
import sys
import wfdb
import numpy as np
import scipy.signal as sg


def read_header(fname):
    """
    Funcion para leer el header de las señales
    """
    
    idx_point = fname.find('.')
    fname = fname[:idx_point]
    header = wfdb.rdheader(fname)
    
    return header
 

#Función para leer las señales

def read_challenge_mat_files(fname):
    """
    Lee las señales .mat del challenge 2017 y devuelve un array de numpy

    Input
    ------
       .- fname: nombre del fichero para leer

    Output
    ------
       .- ecg: numpy array con los valores de ecg
       .- header: dicccionario con la información del header
    """

    #get only the id, remove the extension file
    #fname_header = fname

    #
    #header = read_header(fname_header)

    #read mat file
    header = read_header(fname)

    mat_file = io.loadmat(fname)

    #mat_file is a dictionary, and the data is on the 'var' key

    ecg = mat_file['val'].flatten()

   #check info in header
    #for item in header:
        #print item, ":", header[item]

    return ecg,header.__dict__

def processing_ecg(ecg,fs = 300):
    """
    Function to filter ecg. We are going to make a simple filter between 
    
    """
    
    order = int(0.3*fs)
    fnyq = fs/2.
    fc = [2/fnyq,45/fnyq] #frecuencias de corte 2Hz y 45Hz 
    
    
    a,b =np.array([1]), sg.firwin(order,fc,pass_zero = False) 
    ecg_filtered = sg.filtfilt(b, a, ecg)
    
    return ecg_filtered

#to modify
"""
def deep_conv_lstm_net():

    model = Sequential()
    model.add(Reshape(input_shape = (dim_length,dim_channels), target_shape =(dim_length,dim_channels,1)

    for i in range(9):
        model.add(Convolution2D(64,kernel_size=(3,1),padding='same',kernel_regularizer = 12(0.01)))
        model.add(Activation('relu'))

    model.add(Reshape(target_shape= (dim_length,64*dim_channels)))

    for i in range(3):
        model.add(LSTM(units=64,return_sequences=True,activation = 'tanh'))

    model.add(Dropout(0.5))
    model.add(TimeDistributed(Dense(units=output_dim)))
    model.add(Activation("softmax"))
    model.add(Lambda(lambda x: x[:,-1,:], output_shape[output_dim]))
    model.compile(loss='categorical_crossentropy',optimizer =Adam(lr=0.001), metrics = ['accuracy'])
"""

#if __name__ == "__main__":

#    read_challenge_mat_files(sys.argv[1])
