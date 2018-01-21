#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 11:52:18 2018

@author: obarquero
"""

import scipy.io as io
import sys
import wfdb
import biosppy.signals as sg


mat_file = io.loadmat('A00001.mat')
ecg_file = np.array(mat_file['val']).flatten()

a = sg.ecg.ecg(np.array(ecg_file))