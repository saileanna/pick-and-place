from __future__ import print_function

import mayavi.mlab as mlab
#from grasp_generator.utils.utils import *
from grasp_generator.utils import utils
from grasp_generator.utils.sample import *
import numpy as np
import trimesh

import open3d as o3d


def get_color_plasma_org(x):
    import matplotlib.pyplot as plt
    return tuple([x for i, x in enumerate(plt.cm.plasma(x)) if i < 3])


def get_color_plasma(x):
    return tuple([float(x), float(1 - x), float(0)])


def plot_mesh(mesh):
    assert type(mesh) == trimesh.base.Trimesh
    mlab.triangular_mesh(mesh.vertices[:, 0],
                         mesh.vertices[:, 1],
                         mesh.vertices[:, 2],
                         mesh.faces,
                         colormap='Blues')


def display_pointcloud(xyz, point_size=2) -> None:

    xyz = np.nan_to_num(xyz).reshape(-1, 3)

    point_cloud_open3d = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz))

    visualizer = o3d.visualization.Visualizer()  # pylint: disable=no-member
    visualizer.create_window()
    visualizer.add_geometry(point_cloud_open3d)
    
    visualizer.get_render_option().point_size = point_size
    visualizer.get_render_option().show_coordinate_frame = True
    visualizer.get_view_control().set_front([0, 0, -1])
    visualizer.get_view_control().set_up([0, -1, 0])

    visualizer.run()
    visualizer.destroy_window()


def if_angle_in_range(angle, allowed_range) -> bool:
    
    min_val = allowed_range[0]
    max_val = allowed_range[1]    
    
    if angle <= 0:
        angle += 360
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
    # first iteration preserves not rotation direction
    for angle_name, angle_rad in zip(['alphas', 'betas', 'gammas'], angles):
        angle_degree = int(angle_rad)

        if angle_degree < 0:
            angle_degree += 360
 
        angle_in_range = if_angle_in_range(angle_degree, args[angle_name])
        if not angle_in_range:
            return False

    return True

   
def draw_scene(args,
               pc,
               grasps=[],
               grasp_scores=None,
               grasp_color=None,
               gripper_color=(0, 1, 0),
               mesh=None,
               show_gripper_mesh=False,
               grasps_selection=None,
               visualize_diverse_grasps=False,
               min_seperation_distance=0.03,
               pc_color=None,
               plasma_coloring=False,
               target_cps=None,
               visualize_best_only=False,
               max_grasps=100):
    """
    Draws the 3D scene for the object and the scene.
    Args:
      pc: point cloud of the object
      grasps: list of 4x4 numpy array indicating the transformation of the grasps.
        grasp_scores: grasps will be colored based on the scores. If left 
        empty, grasps are visualized in green.
      grasp_color: if it is a tuple, sets the color for all the grasps. If list
        is provided it is the list of tuple(r,g,b) for each grasp.
      mesh: If not None, shows the mesh of the object. Type should be trimesh 
         mesh.
      show_gripper_mesh: If True, shows the gripper mesh for each grasp. 
      grasp_selection: if provided, filters the grasps based on the value of 
        each selection. 1 means select ith grasp. 0 means exclude the grasp.
      visualize_diverse_grasps: sorts the grasps based on score. Selects the 
        top score grasp to visualize and then choose grasps that are not within
        min_seperation_distance distance of any of the previously selected
        grasps. Only set it to True to declutter the grasps for better
        visualization.
      pc_color: if provided, should be a n x 3 numpy array for color of each 
        point in the point cloud pc. Each number should be between 0 and 1.
      plasma_coloring: If True, sets the plasma colormap for visualizting the 
        pc.
    """


    if pc_color is None and pc is not None:
        if plasma_coloring:
            mlab.points3d(pc[:, 0],
                          pc[:, 1],
                          pc[:, 2],
                          pc[:, 2],
                          colormap='plasma')
        else:
            mlab.points3d(pc[:, 0],
                          pc[:, 1],
                          pc[:, 2],
                          color=(0.1, 0.1, 1),
                          scale_factor=0.01)
    elif pc is not None:
        if plasma_coloring:
            mlab.points3d(pc[:, 0],
                          pc[:, 1],
                          pc[:, 2],
                          pc_color[:, 0],
                          colormap='plasma')
        else:
            rgba = np.zeros((pc.shape[0], 4), dtype=np.uint8)
            rgba[:, :3] = np.asarray(pc_color)
            rgba[:, 3] = 255
            src = mlab.pipeline.scalar_scatter(pc[:, 0], pc[:, 1], pc[:, 2])
            src.add_attribute(rgba, 'colors')
            src.data.point_data.set_active_scalars('colors')
            g = mlab.pipeline.glyph(src)
            g.glyph.scale_mode = "data_scaling_off"
            g.glyph.glyph.scale_factor = 0.01

    grasp_pc = np.squeeze(utils.get_control_point_tensor(1, False), 0)
    grasp_pc[2, 2] = 0.059
    grasp_pc[3, 2] = 0.059

    mid_point = 0.5 * (grasp_pc[2, :] + grasp_pc[3, :])

    modified_grasp_pc = []
    modified_grasp_pc.append(np.zeros((3, ), np.float32))
    modified_grasp_pc.append(mid_point)
    modified_grasp_pc.append(grasp_pc[2])
    modified_grasp_pc.append(grasp_pc[4])
    modified_grasp_pc.append(grasp_pc[2])
    modified_grasp_pc.append(grasp_pc[3])
    modified_grasp_pc.append(grasp_pc[5])

    grasp_pc = np.asarray(modified_grasp_pc)
   

    print('draw scene ', len(grasps))

    selected_grasps_so_far = []
    removed = 0

    if grasp_scores is not None:
        min_score = np.min(grasp_scores)
        max_score = np.max(grasp_scores)


    if visualize_best_only:
        g = grasps[0]

        gripper_color = (0.0, 1.0, 0.0)
        
        pts = np.matmul(grasp_pc, g[:3, :3].T)
        pts = np.matmul(grasp_pc, np.identity(3))
        
        np.save('pts_identity.npy', pts)
        
        pts = np.matmul(grasp_pc, g[:3, :3].T) 
        pts += np.expand_dims(g[:3, 3], 0)        
        np.save('pts_g.npy', pts)
        print(f'\n[visualization_utils] Grasp info of best Grasp: \n{pts}')

        tube_radius = 0.001
        mlab.plot3d(pts[:, 0],
                    pts[:, 1],
                    pts[:, 2],
                    color=gripper_color,
                    tube_radius=tube_radius,
                    opacity=1)
    else:
        wrong_counter  = 0
        true_counter  = 0
        for i in range(len(grasps)):
            g = grasps[i]

            # assume gripper_color is not a list and grasp_scores is not None
            normalized_score = (grasp_scores[i] -
                                min_score) / (max_score - min_score + 0.0001)
            if grasp_color is not None:
                gripper_color = grasp_color[i]
            else:
                gripper_color = get_color_plasma(normalized_score)

            if min_score == 1.0:
                gripper_color = (0.0, 1.0, 0.0)

            pts = np.matmul(grasp_pc, g[:3, :3].T)
            pts += np.expand_dims(g[:3, 3], 0)

            tube_radius = 0.001
            mlab.plot3d(pts[:, 0],
                        pts[:, 1],
                        pts[:, 2],
                        color=gripper_color,
                        tube_radius=tube_radius,
                        opacity=1)

    print('removed {} similar grasps'.format(removed))


def get_axis():
    # hacky axis for mayavi
    axis = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    axis_x = np.array([np.linspace(0, 0.10, 50), np.zeros(50), np.zeros(50)]).T
    axis_y = np.array([np.zeros(50), np.linspace(0, 0.10, 50), np.zeros(50)]).T
    axis_z = np.array([np.zeros(50), np.zeros(50), np.linspace(0, 0.10, 50)]).T
    axis = np.concatenate([axis_x, axis_y, axis_z], axis=0)
    return axis