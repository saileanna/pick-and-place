#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 07:23:12 2022

@author: student
"""
from copy import copy

import numpy as np
from pyniryo import NiryoRobot, PoseObject
 
from base.camera import IntelImage

# Ip-address Ned
robot_ip_address = '10.122.199.240'

class NiryoNed():
    def __init__(self, global_config) -> None:
        
        # Connecting to robot
        self.niryo_robot = NiryoRobot(robot_ip_address)
        print('Connecting was successful')
        # Calibrate robot if robot needs calibration
        self.niryo_robot.calibrate_auto()
        print('\nCalibration was successful')
        # Reduce speed
        self.niryo_robot.set_arm_max_velocity(60)
        # Instantiate global config
        self.global_config = global_config
        # Instantiate camera
        self.cam = IntelImage(global_config)
           
    
    def move_to_pose(self, pose):
        # Limited the range of joint 4, that it couldnt rotate over 97.4 degree
        if abs(pose.roll) > 1.7:
            print('\nERROR: Grasp inside the exclusion zone')
            return 0
        
        else:
            self.niryo_robot.move_pose(pose)
    
        
    def take_picture(self):
        # Define capture pose
        capture_pose = PoseObject(
            x= 0.123, y= -0.006, z= 0.217,
            roll=0.00, pitch= 0.474, yaw= -0.053
            )
        # Move to capture pose
        self.move_to_pose(capture_pose)
        
        # Image capture
        data = self.cam.capture(
            n_frames=self.global_config['n_frames'] + self.global_config['wait_frames'],
            wait_frames=self.global_config['wait_frames']
        )
        
        return data
                  
    
    def pick(self, pose):
        # Pick
        # Offset according to Z-Axis to go over pick pose
        height_offset_z = 0.05
        gripper_speed = 300
        
        # Compute pick pose / pick pose + z position
        pick_pose = copy(pose)
        pick_pose_z = pick_pose.copy_with_offsets(z_offset=height_offset_z)
        
        # Open tool
        self.niryo_robot.open_gripper(gripper_speed)
        
        # Move pick pose + z
        self.move_to_pose(pick_pose_z)
        
        # Move pick pose with reduced speed
        self.niryo_robot.set_arm_max_velocity(10)
        self.move_to_pose(pick_pose)  
        
        # Close tool
        self.niryo_robot.close_gripper(gripper_speed)
        
        # Move pick pose + z
        self.niryo_robot.set_arm_max_velocity(60)
        self.move_to_pose(pick_pose_z)
            
        return True
    
    
    def place(self, pose):
        # Place
        # Offset according to Z-Axis to go over place pose
        height_offset_z = 0.05
        gripper_speed = 300
        
        # Compute place pose / place pose + z position
        place_pose = copy(pose)
        place_pose_z = place_pose.copy_with_offsets(z_offset=height_offset_z)
        
        # Move place pose + z position
        self.move_to_pose(place_pose_z)
        
        # Move place pose with reduced speed
        self.niryo_robot.set_arm_max_velocity(10)
        self.move_to_pose(place_pose)
        
        # Open tool
        self.niryo_robot.open_gripper(gripper_speed)
        
        # Move place pose + z
        self.niryo_robot.set_arm_max_velocity(60)
        self.move_to_pose(place_pose_z)
        
        # Move to home position
        self.niryo_robot.move_to_home_pose()
        
        return True
    
        
    def tear_down(self):
        # Releasing connection
        close = self.niryo_robot.close_connection()
        print('\nDisconnected from Niryo Robot')