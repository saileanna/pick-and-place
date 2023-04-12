# -*- coding: utf-8 -*-
"""
Created on Tue Sep  6 15:55:13 2022

@author: as104
"""

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from pyniryo import PoseObject

from base import plotting
import base.functions as functions
from base.niryo_ned import NiryoNed

import mayavi.mlab as mlab
from grasp_generator.utils.visualization_utils import *

from grasp_generator.grasp import generate_grasps
from grasp_generator.utils import grasp_selection

ROOT_PATH = os.path.abspath('../')
CONFIG_PATH = os.path.join(ROOT_PATH, 'configs')


def main(): 
    # Load global.json
    global_config = functions.load_json(os.path.join(CONFIG_PATH, 'global.json'))    
          
    # Instantiate roboter
    robot = NiryoNed(global_config)
       
    # Take picture
    data = robot.take_picture()
    print('\nCapture of image was successful')
    
    if global_config['generate']:
        # Generate grasp translation and angles in rad
        good_grasps, good_scores, pc, pc_colors = generate_grasps(global_config, data)
        
        if len(good_grasps) == 0:
            print(f'[grasp] No grasp found repeat')
            return 0
               
        else:
            g = good_grasps[0]
            score = good_scores[0]
            
            np.save('g', g, allow_pickle=True)
            np.save('good_grasps', np.array(good_grasps), allow_pickle=True)
            np.save('score', score, allow_pickle=True)
            np.save('pc', pc, allow_pickle=True)
            np.save('pc_colors', pc_colors, allow_pickle=True)
    else:
        # Load grasp translation and angles in rad
        g = np.load('g.npy', allow_pickle=True)
        good_grasps = np.load('good_grasps.npy', allow_pickle=True)
        good_scores = np.load('score.npy', allow_pickle=True)
        pc = np.load('pc.npy', allow_pickle=True)
        pc_colors = np.load('pc_colors.npy', allow_pickle=True)
        
    # Transformation grasp
    angles_and_translations_r = grasp_selection.get_pose(g)
    print(f'\n[main] g: \n{g}')
    print(f'\n[main] angles_and_translations_r : \n{angles_and_translations_r}')
    
    # Visualize grasp
    mlab.figure(bgcolor=(1, 1, 1))
    draw_scene(
        global_config,
        pc,
        pc_color=pc_colors,
        grasps=np.array(good_grasps),
        grasp_scores=np.array(good_scores),
        max_grasps=global_config['display_grasps'],
        visualize_best_only=global_config['only_best']
    )
    print('\nclose the window to continue . . .')
    mlab.show(stop=True)     
   
    # Define pick pose
    pick_pose = PoseObject(*angles_and_translations_r)
    
    # Define place pose
    place_pose = PoseObject(
        x=angles_and_translations_r[0], y=0.117, z=angles_and_translations_r[2],
        roll=angles_and_translations_r[3], pitch=angles_and_translations_r[4], yaw=angles_and_translations_r[5]
        )
    
    if global_config['use_robot']:
        # Pick
        robot.pick(pick_pose)
        
        # Wait 5 sec 
        time.sleep(5)
        
        # Place
        robot.place(place_pose)
    
    else:
        # Releasing connection
        robot.tear_down()
        
        
if __name__ == '__main__':
     main()