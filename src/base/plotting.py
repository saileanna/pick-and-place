#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 15:02:02 2022

@author: student
"""

import os
import numpy as np
import open3d as o3d

from matplotlib import pyplot as plt

ROOT_PATH = os.path.abspath('../')
IMAGE_PATH = os.path.join(ROOT_PATH, 'images')

if not os.path.exists(IMAGE_PATH):
    os.mkdir(IMAGE_PATH)

plt.rcParams["figure.figsize"] = (16, 12)


def display_pointcloud(xyz, point_size=1, savename=None):
    # Visualize pc with open3d
    xyz = np.nan_to_num(xyz).reshape(-1, 3)

    point_cloud_open3d = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz))
    visualizer = o3d.visualization.Visualizer()  # pylint: disable=no-member
    visualizer.create_window(visible=False)
    visualizer.add_geometry(point_cloud_open3d)
    visualizer.get_render_option().point_size = point_size
    visualizer.get_render_option().show_coordinate_frame = True
    visualizer.get_view_control().set_front([0, 0, -1])
    visualizer.get_view_control().set_up([0, -1, 0])
    
    if savename is not None:
        fig, ax = plt.subplots()
        img = visualizer.capture_screen_float_buffer(True)
        plt.imshow(np.asarray(img))
        plt.savefig(os.path.join(IMAGE_PATH, savename))
    else:
        visualizer.run()

    visualizer.destroy_window()


def imshow(val, title=None, colorbar_label=None, xlabel=None, ylabel=None, savename=None):
    # Visualize images
    fig, ax = plt.subplots()

    im = ax.imshow(
        val,
        vmin=np.nanmin(val),
        vmax=np.nanmax(val)
    )
    
    if colorbar_label is not None:
        plt.colorbar(im, label=colorbar_label)
    else:
        plt.colorbar(im)
    ax.grid()
    
    if title is not None:
        ax.set_title(title, fontsize=20)
    
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=15)
        
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=15)

    if savename is not None:
        plt.savefig(os.path.join(IMAGE_PATH, savename))
          
    else:
        plt.show()


def scatter(values, title=None, xlabel=None, ylabel=None, y_scale='linear', savename=None):
    # Visualize scatter
    fig, ax = plt.subplots()

    if isinstance(values, list):
        for val in values:
            plt.scatter(x=range(len(values)), y=val)
    elif isinstance(values, dict):
        for name, val in values.items():
            plt.scatter(x=range(len(val)), y=val, label=name, s=1)
    else:
        plt.scatter(x=range(len(values)), y=values)

    ax.set_yscale(y_scale)
    ax.grid()
    
    if title is not None:
        ax.set_title(title, fontsize=20)
    
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=15)

    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=15)

    if isinstance(values, dict):
        plt.legend(loc='best', fontsize=15)
        
    if savename is not None:
        plt.savefig(os.path.join(IMAGE_PATH, savename))
    
    else:
        plt.show()


def histo(val, dim=2, vline=None, bins=100, xlabel=None, ylabel=None, title=None, savename=None):
    # Visualize histogram
    fig, ax = plt.subplots()

    if isinstance(val, dict):
        for name, v in val.items():
            ax.hist(v.reshape(-1,1), bins=bins, alpha=0.5, label=name)   
             
    else:
        ax.hist(val.reshape(-1,1), bins=bins)   
        
    
    if isinstance(val, dict):
        plt.legend(loc='best', fontsize=15)
        
    if vline is not None:
        if isinstance(vline, list):
            for line in vline:
                ax.axvline(x=line, lw=2, color='r')
                
        else:
            ax.axvline(x=vline, lw=2, color='r')
            
    ax.grid()
            
    if title is not None:
        ax.set_title(title, fontsize=20)
        
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=15)
        
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=15)
    
    if savename is not None:
        plt.savefig(os.path.join(IMAGE_PATH, savename))
        
    else:
        plt.show()
    
    
# Define global variable for the following function
CUT_BORDER_LIST_H = []


def display_imshow_cropborder_H(depth_img, title=None, xlabel=None, ylabel=None) -> float:
    # Display imshow horizontal border    
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)

    if title:
        ax.set_title(title, fontsize=20)
        
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=15)
        
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=15)
        
    ax.imshow(depth_img)

    ax.grid()
    
    cur_lines = []
    
    
    def onclick(event):
        
        # remove last rect if exists
        if len(cur_lines) > 1:
            cur_lines[0].remove()
            del cur_lines[0]

        global ix, iy
        ix, iy = event.xdata, event.ydata

        cur_line = ax.axhline(iy, c='r')

        # add and plot rect as patch
        cur_lines.append(cur_line)

        global CUT_BORDER_LIST_H
        CUT_BORDER_LIST_H.append(int(iy))

        fig.canvas.draw()

    cid = fig.canvas.mpl_connect('button_press_event', onclick)

    plt.show()

    if len(CUT_BORDER_LIST_H) < 2:
        print('[Horizontal] Error, too few values, choose 2, exit')
        return None
    return CUT_BORDER_LIST_H[-2:]

# Define global variable for the following function
CUT_BORDER_LIST_V = []


def display_imshow_cropborder_V(depth_img, title=None, xlabel=None, ylabel=None) -> float:
    # Display imshow vertical border   
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)

    if title:
        ax.set_title(title, fontsize=20)
        
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=15)
        
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=15)

    ax.imshow(depth_img)

    ax.grid()
    
    cur_lines = []
    
    
    def onclick(event):
        
        # Remove last rect if exists
        if len(cur_lines) > 1:
            cur_lines[0].remove()
            del cur_lines[0]

        global ix, iy
        ix, iy = event.xdata, event.ydata

        cur_line = ax.axvline(ix, c='r')

        # Add and plot rect as patch
        cur_lines.append(cur_line)

        global CUT_BORDER_LIST_V
        CUT_BORDER_LIST_V.append(int(ix))
        
        fig.canvas.draw()

    cid = fig.canvas.mpl_connect('button_press_event', onclick)

    plt.show()

    if len(CUT_BORDER_LIST_V) < 2:
        print('[Vertical] Error, too few values, choose 2, exit')
        return None
    
    return CUT_BORDER_LIST_V[-2:]