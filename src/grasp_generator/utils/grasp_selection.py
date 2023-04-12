#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 09:52:17 2022

@author: student
"""

import numpy as np
import trimesh.transformations as tra
from numpy import cos, sin, arctan2, sqrt


def sort_grasps_scores(g, s):
    grasps_np = np.array(g)
    score_np = np.array(s)

    idx = score_np.argsort()

    return  grasps_np[idx], score_np[idx]


def degree_to_rad(val):
    if isinstance(val, list) or isinstance(val, tuple):
        tmp = []
        for angle in val:
            tmp.append(angle / 180. * np.pi)
        return tmp
    return val / 180. * np.pi


def rad_to_degree(val):
    if isinstance(val, list) or isinstance(val, tuple):
        tmp = []
        for angle in val:
            tmp.append(angle / np.pi * 180.)
        return tmp
    return val / np.pi * 180.


def get_R(alpha, beta, gamma):
    # Compute rotation matrix from angles
    l1 = [
        cos(alpha) * cos(beta),
        sin(alpha) * cos(beta),
        -sin(beta)
    ]
    l2 = [
        cos(alpha) * sin(beta) * sin(gamma) - sin(alpha) * cos(gamma),
        sin(alpha) * sin(beta) * sin(gamma) + cos(alpha) * cos(gamma),
        cos(beta) * sin(gamma)
    ]
    l3 = [
        cos(alpha) * sin(beta) * cos(gamma) + sin(alpha) * sin(gamma),
        sin(alpha) * sin(beta) * cos(gamma) - cos(alpha) * sin(gamma),
        cos(beta) * cos(gamma)
    ]
    
    return np.array([l1,l2,l3])


def create_angle_list_from_range(min_val, max_val, step_size=10):
    if min_val > max_val:
        return list(range(min_val, 360, step_size)) + list(range(0, max_val + step_size, step_size))
    return list(range(min_val, max_val + step_size, step_size))


def rotate_by_angles(se4, angles):
    se4[:3, :3] = np.matmul(se4.copy()[:3,:3], R)
    return se4


def rotate_by_R(se4, R):
    se4[:3, :3] = np.matmul(se4.copy()[:3,:3], R[:3,:3])
    return se4


def correct_angle(val):
    if val < 0:
        val += 2 * np.pi
    return val


def angles_from_R(R):
    # Compute roll, pitch and yaw from rotation matrix
    phi_x = correct_angle(arctan2(R[2,1],R[2,2]))
    phi_y = correct_angle(arctan2(-R[2,0], sqrt(R[2,1]*R[2,1] + R[2,2]*R[2,2])))
    phi_z = correct_angle(arctan2(R[1,0],R[0,0]))

    return (phi_x, phi_y, phi_z)

    alpha = arctan2(R[1,0],R[0,0])
    beta = arctan2(-R[2,0], sqrt(R[2,1]*R[2,1] + R[2,2]*R[2,2]))
    gamma = arctan2(R[2,1],R[2,2])

    return (alpha, beta, gamma)


def euler_from_matrix(g, mode='sxyz') -> tuple:
    # Compute roll, pitch and yaw from rotation matrix in xyz fixed angles
    return tra.euler_from_matrix(g[:3, :3], mode)


def get_pose(g):
    # Transformation grasp
    # Offset camera to gripper
    T_cam = np.array([-0.01628403,  0.02678967,  0.04926566])
    # Offset Ned base to gripper
    T_ned = np.array([0.123, -0.006, 0.217])
    
    # Rotation angles gripper in capture pose
    alpha = 0.000
    beta = 0.474
    gamma = -0.053

    ned_euler = np.array([alpha, beta, gamma])
    # Rotation matrix gripper in capture pose
    R_ned = tra.euler_matrix(*ned_euler, axes='sxyz')
    
    # Transformation matrix
    A = np.array([
        [0, -1, 0],
        [0, 0, -1],
        [1, 0, 0]
    ])

    #I = np.identity(3)
    #I[1, 1] = -1
    
    #A_inv_y = np.matmul(I, A)
    
    # Transformation rotation
    R_g_euler = np.array(tra.euler_from_matrix(g[:3, :3], 'sxyz'))
    R_g_euler_A = np.matmul(R_g_euler, A)
    R_part = (R_g_euler_A + ned_euler)
    
    # Transformation translation
    T_part = np.matmul(np.matmul(g[:3, 3], A) + T_cam, R_ned[:3, :3].T) + T_ned
    #K_correctur = np.array([0.015, 0.0, 0.0])
    #T_part += K_correctur
    
    # Fully grasp transformation
    translations_angles_r = np.append(T_part, R_part)
    
    return (translations_angles_r)
    

def if_angle_in_range(name, angle, allowed_range) -> bool:
    # Allow angle input -20 degree instead of 240
    min_val = allowed_range[0]
    max_val = allowed_range[1]    

    if min_val < 0:
        min_val += 360
    if max_val <= 0:
        max_val += 360

    if min_val > max_val:
        if (angle + 1 >= min_val and angle - 1 <= 360) or (angle >= 0 and angle - 1 <= max_val):
            
            return True 
        
        return False
    
    if angle + 1 >= min_val and angle - 1 <= max_val:
        
        return True
    
    return False


def if_angles_in_range(args, angles):
    # Angle limitation
    for angle_name, angle_rad in zip(['alphas', 'betas', 'gammas'], angles):

        angle_deg = rad_to_degree(angle_rad)
        
        angle_degree = int(angle_deg)
        
        if angle_degree < 0:
            angle_degree += 360
        
        angle_in_range = if_angle_in_range(angle_name, angle_degree, args[angle_name])
        if not angle_in_range:
            return False

    return True


def if_pos_in_range(pos):
    # Minimum height
    if pos[1] > 0.05:
        return False
    return True


def apply_restrictions(args, pos, angles):
    # Apply restrictions
    if not if_pos_in_range(pos):
        return False
    if not if_angles_in_range(args, angles):
        return False
    return True    


def gen_angles(args):
    angles_list = []

    if args['gen_alpha']:
        angle_name = args['alphas']
    elif args['gen_beta']:
        angle_name = args['betas']
    elif args['gen_gamma']:
        angle_name = args['gammas']

    min_val = angle_name[0]
    max_val = angle_name[1]
    
    if min_val > max_val:
        return 360 - (args['step_size'] * round(((np.random.randint(min_val, 360+max_val) % 360) /  args['step_size'])))
    return 360 - (args['step_size'] * round((np.random.randint(min_val, max_val) / args['step_size'])))


def print_grasp_information(gs, ss):
    for g, s in zip(gs, ss):
        R = g[:3,:3]
        X, Y, Z = g[:3,3]

        angles = angles_from_R(R)