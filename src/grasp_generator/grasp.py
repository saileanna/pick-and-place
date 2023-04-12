from __future__ import print_function

import sys
import os
import glob
import time

import numpy as np
import mayavi.mlab as mlab
from grasp_generator.utils.visualization_utils import *

from grasp_generator import grasp_estimator
from grasp_generator.utils import utils
from grasp_generator.utils.visualization_utils import *
import grasp_generator.utils.grasp_selection as grasp_selection
from base import plotting, functions
import trimesh.transformations as tra

ROOT_PATH = os.path.abspath('../')
SAVINGS_PATH = os.path.join(ROOT_PATH, 'savings')
IMAGE_PATH = os.path.join(ROOT_PATH, 'images')


def get_color_for_pc(pc, K, color_image):
    proj = pc.dot(K.T)
    proj[:, 0] /= proj[:, 2]
    proj[:, 1] /= proj[:, 2]

    pc_colors = np.zeros((pc.shape[0], 3), dtype=np.uint8)
    for i, p in enumerate(proj):
        x = int(p[0])
        y = int(p[1])
        pc_colors[i, :] = color_image[y, x, :]

    return pc_colors


def generate_grasps(args, data):
    grasp_sampler_args = utils.read_checkpoint_args(   
        os.path.join(ROOT_PATH, 'src', 'grasp_generator', args['grasp_sampler_folder'])
    )
    grasp_sampler_args.is_train = False
    grasp_evaluator_args = utils.read_checkpoint_args(
        os.path.join(ROOT_PATH, 'src', 'grasp_generator', args['grasp_evaluator_folder'])
    )
    grasp_evaluator_args.continue_train = True
    estimator = grasp_estimator.GraspEstimator(grasp_sampler_args,
                                               grasp_evaluator_args, args)
    
    depth = data['depth']
    image = data['image']
    K = data['intrinsics_matrix']
    
    # Visualize depth image
    plotting.imshow(depth)
    
    # Removal depth data
    np.nan_to_num(depth, copy=False)
    mask = np.where(np.logical_or(depth == min(args['cut_values']), depth > max(args['cut_values'])))
    depth[mask] = np.nan

    # Backprojection to generate pc 
    pc, selection = functions.backproject(depth,
                                K,
                                return_finite_depth=True,
                                return_selection=True,
                                save_pc_with_infinites=True)
        
    pc_colors = image.copy()
    pc_colors = np.reshape(pc_colors, [-1, 3])
    pc_colors = pc_colors[selection, :]
    
    if args['test_pc_show']:
        object_pc = data['smoothed_object_pc']   
        plotting.display_pointcloud(object_pc, point_size=2)
        mlab.points3d(pc[:, 0],
                      pc[:, 1],
                      pc[:, 2],
                      pc[:, 2],                 
                      colormap='plasma',
                      scale_factor=0.01
                      )
        print('Close the window to continue to next object . . .')
        mlab.show()
        return 

    # Smoothed pc comes from averaging the depth for 10 frames and removing
    # The pixels with jittery depth between those 10 frames.
    object_pc = data['smoothed_object_pc']    
    
    # Origin grasp from camera coordinate system
    original_grasp = np.identity(4)

    good_grasps = []
    good_scores = []
    
    angles_and_translations_r = None
    
    # It generate_test_grasps is not set to true grasps from framework w.r.t object_pc are generated
    # Else num_grasp grasps based on the angles depending on their ranges are generated
    if not args['generate_test_grasps']:
 
        # Generate grasps and their corresponding scores    
        generated_grasps, generated_scores = estimator.generate_and_refine_grasps(object_pc)

        # Sort grasp w.r.t their scores
        generated_grasps_sorted, generated_scores_sorted = grasp_selection.sort_grasps_scores(generated_grasps, generated_scores)

        # Visualize histo
        #plotting.histo(generated_scores_sorted, bins=100, xlabel='Score', ylabel='Häufigkeit', title=None, savename='score')

        # Save generated grasps
        np.save('generated_grasps_sorted', generated_grasps_sorted, allow_pickle=True)
        
        # Iterate over all grasps and check if their angle conditions are met (begin with these grasps with best scores)
        for g, s in zip(generated_grasps_sorted, generated_scores_sorted):
            
            # Extract angles (alpha, beta, gamma) from grasp rotation matrix
            g_trans =  g[:3,3]
            g_angles =  tra.euler_from_matrix(g[:3,:3])  
            
            # Check is angles meet the conditions
            if grasp_selection.apply_restrictions(args, pos=g[:3, 3], angles=g_angles):
                good_grasps.append(g)
                good_scores.append(s)
   
        print(f'[grasp] len of good_grasps {len(good_grasps)} len of all_grasps ({len(generated_grasps_sorted)})')    
        return good_grasps, good_scores, pc, pc_colors
        
            
        print('\nPrint information of best grasp:\n\tangles: '.format(
            grasp_selection.rad_to_degree(
                grasp_selection.euler_from_matrix(
                    good_grasps[0]
                    )
                )
            )
        )
        g = good_grasps[0]
        a = grasp_selection.euler_from_matrix(good_grasps[0])
        b = grasp_selection.rad_to_degree(a)

    else:
        # Generate list of valid angles [alphas, betas, gammas]
        good_grasps.append(original_grasp)
        good_scores.append(0.75)
    
        for _ in range(args['num_test_grasps']):
            rotation_angles = [0., 0., 0.]
            for idx, gen_angle in enumerate(['gen_alpha', 'gen_beta', 'gen_gamma']):
                if args[gen_angle]:
                    u = args['betas']
                    rotation_angles[idx] = grasp_selection.gen_angles(args)
            
            R = grasp_selection.get_R(
                grasp_selection.degree_to_rad(rotation_angles[2]),
                grasp_selection.degree_to_rad(rotation_angles[1]),
                grasp_selection.degree_to_rad(rotation_angles[0])
            )

            dummy_grasp = grasp_selection.rotate_by_R(
                original_grasp.copy(),
                R
            )
            
            good_grasps.append(dummy_grasp)
            good_scores.append(0.2)
   
    # Visualize grasp
    if args['visualize']:
        if len(good_grasps) == 0:
            print('\nLength of good grasps is zero!')
            return None
        mlab.figure(bgcolor=(1, 1, 1))
        draw_scene(
            pc,
            pc_color=pc_colors,
            grasps=np.array(good_grasps),
            grasp_scores=np.array(good_scores),
            max_grasps=args['display_grasps'],
            visualize_best_only=args['only_best']
        )
        print('\nclose the window to continue . . .')
        mlab.show(stop=True)            
    
    translation_and_rotation = grasp_selection.get_pose(good_grasps[0])
    
    return grasp_selection.get_pose(good_grasps[0]), pc, pc_color, grasps,  grasp_scores