#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 09:49:48 2022

@author: student
"""

import os
import json

import cv2 as cv
import numpy as np
import open3d as o3d

from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score

from matplotlib import pyplot as plt
import matplotlib.patches as patches

from base import plotting
from base.typings import SampleArea

plt.rcParams["figure.figsize"] = (16, 12)

ROOT_PATH = os.path.abspath('../')
SAVINGS_PATH = os.path.join(ROOT_PATH, 'savings')


def load_json(data_path):
    try:
        with open(data_path, 'r') as f:
            return json.load(f)
    except:
        print(f'[utils] Error loading {os.path.basename(data_path)} from {os.path.dirname(data_path)}')
        return None
    

def save_json(val, data_path):
    with open(data_path, 'w') as f:
        f.write(val)


def remove_nans(sample):
    return sample[~np.isnan(sample).any(axis=1)]


def backproject(depth_cv,
                intrinsic_matrix,
                return_finite_depth=False,
                return_selection=False,
                save_pc_with_infinites=False):

    depth = depth_cv.astype(np.float32, copy=True)
    
    # Get intrinsic matrix
    K = intrinsic_matrix
    Kinv = np.linalg.inv(K)

    # Compute the 3D points
    width = depth.shape[1]
    height = depth.shape[0]

    # Construct the save_pc_with_infinites 2D points matrix
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    ones = np.ones((height, width), dtype=np.float32)
    x2d = np.stack((x, y, ones), axis=2).reshape(width * height, 3)

    # Backprojection
    R = np.dot(Kinv, x2d.transpose())

    # Compute the 3D points
    X = np.multiply(np.tile(depth.reshape(1, width * height), (3, 1)), R)
    X = np.array(X).transpose()

    if save_pc_with_infinites:
        np.save(os.path.join(SAVINGS_PATH, 'camera_R.npy'), R)
        np.save(os.path.join(SAVINGS_PATH, 'camera_depth.npy'), depth)
        np.save(os.path.join(SAVINGS_PATH, 'camera_X.npy'), X)

    if return_finite_depth:
        selection = np.isfinite(X[:, 0])
        X = X[selection, :]

    if return_selection:
        return X, selection
    
    return X


def sample_from_area(sample, borders, nan_removing=True):
    train = sample[borders.ymin:borders.ymax,borders.xmin:borders.xmax,:2].reshape(-1,2)
    test = sample[borders.ymin:borders.ymax,borders.xmin:borders.xmax,2].reshape(-1,1)
    if nan_removing:
        return [remove_nans(train), remove_nans(test)]
    return [train, test]


def create_bkg_mask(pc, model, threshold):
    # Create background mask
    mask_background = np.zeros_like(pc[:,:,0]).astype(bool)    

    for row_idx in range(pc.shape[0]):
        for colum_idx in range(pc.shape[1]):
            point = pc[row_idx, colum_idx]
            if np.isnan(point).any():
                continue
            z_pred = model.predict(np.expand_dims(point[:2], axis=1).transpose())

            if abs(z_pred - point[2]) > threshold:
                mask_background[row_idx, colum_idx] = True
                
    return mask_background


def remove_bkg(global_config, depth, K):
    
    # Create point cloud out of depth image and reshape it
    pc_no_refine = backproject(depth, K, return_finite_depth=False)
    pc_reshaped = pc_no_refine.reshape(*depth.shape, 3)
    
    # Create area for train and test set 
    train_area_path = os.path.join(ROOT_PATH, 'configs', 'train_area.json')
    test_area_path = os.path.join(ROOT_PATH, 'configs', 'test_area.json')
    
    if global_config['define_areas_for_training']:
        # Train area
        train_vlines = plotting.display_imshow_cropborder_V(depth, title='Define two vertical lines for training set')
        train_hlines = plotting.display_imshow_cropborder_H(depth, title='Define two horizonal lines for training set')
        if train_vlines is not None and train_hlines is not None:
            train_area = SampleArea(xmin=min(train_vlines), xmax=max(train_vlines), ymin=min(train_hlines), ymax=max(train_hlines))
            # Save
            save_json(train_area.toJSON(), train_area_path)
        else:
            return None
        # Test area
        test_vlines = plotting.display_imshow_cropborder_V(depth, title='Define two vertical lines for test set')
        test_hlines = plotting.display_imshow_cropborder_H(depth, title='Define two horizonal lines for test set')

        if test_vlines is not None and test_hlines is not None:
            test_area = SampleArea(xmin=min(test_vlines), xmax=max(test_vlines), ymin=min(test_hlines), ymax=max(test_hlines))
            # Save
            save_json(test_area.toJSON(), test_area_path)
        else:
            return None
    else:
        train_dict = load_json(train_area_path)
        test_dict = load_json(test_area_path)
        train_area = SampleArea(**train_dict)
        test_area = SampleArea(**test_dict)
        
    # Obtain train and test data
    X_train, y_train = sample_from_area(pc_reshaped, train_area, nan_removing=True)
    X_test, y_test = sample_from_area(pc_reshaped, test_area, nan_removing=True)
    
    # Train model
    model = LinearRegression()
    reg = model.fit(X_train,y_train)
    
    # Evaluate model
    preds = model.predict(X_test)
    score = reg.score(X_train,y_train)

    # Find max difference
    max_deviation = np.nanmax(np.abs(preds-y_test))
    threshold = global_config['delta'] + max_deviation
    
    # Create background mask
    mask = create_bkg_mask(pc_reshaped, model, threshold)

    # Remove background from original depth image and create pc
    depth_signal = depth * mask
    depth_signal[depth_signal == False] = np.nan

    # Create pc without background
    pc_signal, selection = backproject(
        depth_signal,
        K,
        return_finite_depth=True,
        return_selection=True
    )
    
    return pc_signal