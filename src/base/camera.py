#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  7 11:41:30 2022

@author: student
"""

import os
import sys
import glob

import numpy as np
import pandas as pd
import pyrealsense2 as rs

from base import plotting, functions

    
class IntelImage:
    def __init__(self, global_config) -> None:

        # Global_config
        self.global_config = global_config
        
        # Create a context object. This object owns the handles to all connected realsense devices
        self.pipeline = rs.pipeline()
        
        # Configure depth, infrared and color streams
        self.config = rs.config()    
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.infrared, 640, 480, rs.format.y8, 30)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        
        self.intrinsics = None
        
        
    def hardware_reset(self):
        # Reset camera if camera needs reset
        ctx = rs.context()
        devices = ctx.query_devices()
        for dev in devices:
            print(f'\n{dev} resetted on {id(dev)}')
            dev.hardware_reset()
        

    def bgr_to_rgb(self, image: np.ndarray) -> np.ndarray:
        # Convert bgr to rgb
        return image[:,:,[2,1,0]]
    
    
    def get_intrinsics(self, cfg) -> np.ndarray:
        # Fetch stream profile for color stream
        profile = cfg.get_stream(rs.stream.color)
        # Downcast to video_stream_profile and fetch intrinsics
        intr = profile.as_video_stream_profile().get_intrinsics() 
        # Compute intrinsics camera matrix
        K = np.zeros(shape=(3,3))
        K[0,0] = intr.fx
        K[1,1] = intr.fy
        K[0,2] = intr.ppx
        K[1,2] = intr.ppy
        K[2,2] = 1
        
        self.intrinsics = K   
        
        
    def capture(self, n_frames=10, wait_frames=3) -> dict:            
       # Start streaming
       cfg = self.pipeline.start(self.config)
       
       # Create an align object
       # rs.align allows us to perform alignment of depth frames to others frames
       # The "align_to" is the stream type to which we plan to align depth frames
       align_to = rs.stream.color
       align = rs.align(align_to)
       
       # Enabled HDR
       sensor_dep = self.pipeline.get_active_profile().get_device().first_depth_sensor()
       sensor_dep.set_option(rs.option.hdr_enabled, True)        

       if self.intrinsics is None:
           self.get_intrinsics(cfg)

       depths = np.empty(shape=(480,640,n_frames), dtype=np.float64)

       self.hardware_reset()
       
       df = pd.DataFrame()

       try:
           for frame_it in range(n_frames+wait_frames):
               # Wait for a coherent pair of frames: depth and color
               frames = self.pipeline.wait_for_frames()
               
               # Align the depth frame to color frame
               aligned_frames = align.process(frames)
               
               # Get aligned frames
               aligned_depth_frame = aligned_frames.get_depth_frame() 
               color_frame = aligned_frames.get_color_frame()
               depth_image_aligned = np.asanyarray(aligned_depth_frame.get_data())
               
               if frame_it < wait_frames:
                   frame_it += 1
                   continue

               # Convert images to numpy arrays
               depth_image = np.asanyarray(aligned_depth_frame.get_data(), dtype=np.float64)/ 1000.
               
               # Removal depth data
               mask = np.where(
                   np.logical_or(
                       depth_image == min(self.global_config['cut_values']),
                       depth_image > max(self.global_config['cut_values'])))
               depth_image[mask] = np.nan
               color_image = np.asanyarray(color_frame.get_data())

               if np.nanmax(depth_image) > 5:
                   print(f'np.nanmax(depth_image): {np.nanmax(depth_image)}')
                   continue
               else:
                   df[f'{frame_it-wait_frames}'] = np.squeeze(depth_image.reshape(-1,1))
                   frame_it += 1

       except:
           print('\nFrame didn\"t arrive occurs')
           return None

       finally:
           # Stop streaming
           print('\nstream stopped')
           self.pipeline.stop()

       # Average depth frames   
       df['mean'] = df[[f'{i}' for i in range(n_frames)]].mean(axis=1)
       depth = df['mean'].to_numpy().reshape(480,640)
       smoothed_object_pc = depth
       
       # Remove background from pc
       if self.global_config['remove_background']:
           smoothed_object_pc = functions.remove_bkg(self.global_config, depth, self.intrinsics)
       
       data = {
           "depth": depth,
           "image": self.bgr_to_rgb(color_image),
           "smoothed_object_pc": smoothed_object_pc
       }

       return {**data, "intrinsics_matrix": self.intrinsics}