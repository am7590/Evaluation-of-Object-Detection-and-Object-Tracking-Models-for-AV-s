"""
    Author: Alek Michelson (am7590)
    Class: CSCI 739 (Topics in Artificial Intelligence)
    Project: Evaluation of Object Detection and Object Tracking Models for Autonomous Vehicles

    This script converts spatial data from the CARLA simulator into KITTI-format files for evaluation. 
    It processes transformation files and bounding box data into corresponding matrices, and then 
    converts them into a format suitable for object detection and tracking tasks. The output includes 
    KITTI-format files and calibration data, which are saved to output /kitti_output.

    Running this script assumes you have a /carla data folder, which is included in the git repo.
    Make sure to update FRAME_COUNT if your data has more frames than the example data.
    
    NOTE: This script does not assign object labels in the KITTI output. These are 'DontCare' by default.
"""

import os
import numpy as np
import re

FRAME_COUNT = 300

def ls(path):
    """
    List files and directories at the specified path.
    """
    print("\n****")
    print("Files and directories in '", path, "' :")
    print(os.listdir(path))
    print("\n****")

def read_file(path):
    """
    Reads the content of the specified file into a list of lines.
    """
    file = open(path,"r")
    lines=file.readlines()
    file.close()
    return lines

def get_transform_matrix(x, y, z, pitch, yaw, roll):
    """
    Computes a 4x4 transformation matrix based on translation and rotation values.
    """
    translation_mat = [   
        [1, 0, 0, x],
        [0, 1, 0, y],
        [0, 0, 1, z],
        [0, 0, 0, 1],
    ]

    # assuming yaw is around z-axis
    rot_z = np.identity(4)
    rot_z[0,0] = np.cos((np.pi/180)*yaw)
    rot_z[0,1] = -np.sin((np.pi/180)*yaw)
    rot_z[1,0] = np.sin((np.pi/180)*yaw)
    rot_z[1,1] = np.cos((np.pi/180)*yaw)

    # assuming pitch is around y-axis
    rot_y = np.identity(4)
    rot_y[0,0] = np.cos((np.pi/180)*pitch)
    rot_y[0,2] = -np.sin((np.pi/180)*pitch)
    rot_y[2,0] = np.sin((np.pi/180)*pitch)
    rot_y[2,2] = np.cos((np.pi/180)*pitch)

    # assuming roll is around x-axis
    rot_x = np.identity(4)
    rot_x[1,1] = np.cos((np.pi/180)*roll)
    rot_x[1,2] = np.sin((np.pi/180)*roll)
    rot_x[2,1] = -np.sin((np.pi/180)*roll)
    rot_x[2,2] = np.cos((np.pi/180)*roll)

    # assuming translation is before rotations
    # assuming roations are done in around z,y,x axis (in that order)
    transform_mat = np.matmul(rot_z, rot_y)
    transform_mat = np.matmul(transform_mat, rot_x)
    transform_mat = np.matmul(transform_mat, translation_mat)

    return transform_mat

def process_transform_file(path):
    """
    Processes a file containing transformation data into matrices and their inverses.
    """
    lines = read_file(path)
    mats = []
    mats_inv = []
    for line in lines:
        pattern = r'Location\(x=(.*), y=(.*), z=(.*)\), Rotation\(pitch=(.*), yaw=(.*), roll=(.*)\)\)'
        res = re.search(pattern, line)
        x = float(res.group(1))
        y = float(res.group(2))
        z = float(res.group(3))
        pitch = float(res.group(4))
        yaw = float(res.group(5))
        roll = float(res.group(6))
        mat = get_transform_matrix(x, y, z, pitch, yaw, roll)
        mats.append(mat)
        mats_inv.append(np.linalg.inv(mat))
    return(mats, mats_inv)

def process_bb_file(path):
    """
    Processes a file containing bounding box data into a list of 3D bounding boxes.
    """
    lines = read_file(path)
    bbs = []
    for line in lines:
        pattern = r'Location\(x=(.*), y=(.*), z=(.*)\), Extent\(x=(.*), y=(.*), z=(.*)\),'
        res = re.search(pattern, line)
        x = float(res.group(1))
        y = float(res.group(2))
        z = float(res.group(3))
        ex = float(res.group(4))
        ey = float(res.group(5))
        ez = float(res.group(6))

        x_min = x - ex / 2
        x_max = x + ex / 2
        y_min = y - ey / 2
        y_max = y + ey / 2
        z_min = z - ez / 2
        z_max = z + ez / 2

        points = []
        points.append([x_min, y_min, z_min, 1])
        points.append([x_min, y_min, z_max, 1])
        points.append([x_min, y_max, z_min, 1])
        points.append([x_min, y_max, z_max, 1])
        points.append([x_max, y_min, z_min, 1])
        points.append([x_max, y_min, z_max, 1])
        points.append([x_max, y_max, z_min, 1])
        points.append([x_max, y_max, z_max, 1])

        bbs.append(points)

    return bbs

def apply_transform(bb, mat):
    """
    Applies a transformation matrix to each point of a 3D bounding box.
    """
    bb_new = []
    for pt in bb:
        bb_new.append(np.matmul(mat, np.transpose(pt)))

    return(bb_new)

def get_2d_bbox_from_3d(bb_3d):
    """
    Extracts a 2D bounding box from a 3D bounding box.
    """
    xs = [p[0] for p in bb_3d]
    ys = [p[1] for p in bb_3d]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    return [x_min, y_min, x_max, y_max]

def format_kitti_bbox(class_name, bb_3d, dimensions, location, rotation_y, alpha):
    """
    Formats a 3D bounding box into the KITTI format
    """
    bbox_2d = get_2d_bbox_from_3d(bb_3d)
    kitti_line = create_kitti_format_line(class_name, 0, 0, alpha, bbox_2d, dimensions, location, rotation_y)
    return kitti_line

def create_kitti_format_line(class_name, truncated, occluded, alpha, bbox, dimensions, location, rotation_y):
    """
    Creates a line of text in the KITTI format.
    """
    line = f"{class_name} {truncated} {occluded} {alpha} " \
           f"{bbox[0]:.2f} {bbox[1]:.2f} {bbox[2]:.2f} {bbox[3]:.2f} " \
           f"{dimensions[0]:.2f} {dimensions[1]:.2f} {dimensions[2]:.2f} " \
           f"{location[0]:.2f} {location[1]:.2f} {location[2]:.2f} {rotation_y:.2f}\n"
    return line

def compute_2d_bbox_from_corners(corners):
    """
    Computes a 2D bounding box from the 3D corners of a bounding box.
    """
    xs = [corner[0] for corner in corners]
    ys = [corner[1] for corner in corners]
    return [min(xs), min(ys), max(xs), max(ys)]

def calculate_dimensions(bb):
    """
    Calculates the dimensions of a bounding box.
    """
    x_coords = [point[0] for point in bb]
    y_coords = [point[1] for point in bb]
    z_coords = [point[2] for point in bb]
    length = max(x_coords) - min(x_coords)
    width = max(y_coords) - min(y_coords)
    height = max(z_coords) - min(z_coords)
    return [length, height, width]  

def calculate_centroid(bb):
    """
    Calculates the centroid of a bounding box.
    """
    x_coords = [point[0] for point in bb]
    y_coords = [point[1] for point in bb]
    z_coords = [point[2] for point in bb]
    centroid_x = sum(x_coords) / len(x_coords)
    centroid_y = sum(y_coords) / len(y_coords)
    centroid_z = sum(z_coords) / len(z_coords)
    return [centroid_x, centroid_y, centroid_z]

def estimate_yaw(bb):
    """
    Estimates the yaw of a bounding box.
    """
    dx = bb[1][0] - bb[0][0]
    dy = bb[1][1] - bb[0][1]
    yaw = np.arctan2(dy, dx)  
    return yaw

def calculate_alpha(position, rotation_y):
    """
    Calculates the alpha angle between the position and rotation_y.
    """
    x, z = position[0], position[2]
    alpha = np.arctan2(x, z) - rotation_y
    return alpha

def calculate_camera_intrinsics(fov, img_width, img_height):
    f = img_width / (2 * np.tan(np.deg2rad(fov / 2)))
    cx = img_width / 2
    cy = img_height / 2
    return np.array([
        [f, 0, cx],
        [0, f, cy],
        [0, 0, 1]
    ])

def parse_transform(line):
    """
    Parses a transform line into a list of floating-point numbers.
    """
    numbers = re.findall(r"[-+]?\d*\.\d+|\d+", line)
    return [float(num) for num in numbers]

def load_extrinsic_parameters(pose_file_path):
    """
    Loads extrinsic parameters from file.
    """
    extrinsics = np.eye(4)
    with open(pose_file_path, 'r') as file:
        for line in file:
            if 'Transform' in line:
                values = parse_transform(line)
                # assumes the order is x, y, z, pitch, yaw, roll
                translation = np.array(values[0:3])
                rotation = rotation_matrix_from_euler(values[5], values[4], values[3])
                # extrinsics matrix
                extrinsics[:3, :3] = rotation
                extrinsics[:3, 3] = translation
                break  
    return extrinsics

def rotation_matrix_from_euler(roll, pitch, yaw):
    """
    Returns a rotation matrix from Euler angles.
    I am not sure if there is a better way to do this...
    """
    roll = np.radians(roll)
    pitch = np.radians(pitch)
    yaw = np.radians(yaw)

    R_x = np.array([[1, 0, 0],
                    [0, np.cos(roll), -np.sin(roll)],
                    [0, np.sin(roll), np.cos(roll)]])
    R_y = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                    [0, 1, 0],
                    [-np.sin(pitch), 0, np.cos(pitch)]])
    R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                    [np.sin(yaw), np.cos(yaw), 0],
                    [0, 0, 1]])

    R = np.dot(R_z, np.dot(R_y, R_x))
    return R

def calculate_projection_matrix(K, extrinsics):
    """
    Calculates the camera projection matrix P from intrinsics and extrinsics.
    """
    P = np.dot(K, extrinsics[:3])  # only rotation and translation
    return P

def main():
    data_folder = "./carla/"
    output_folder = "./kitti_output/"
    bbs = process_bb_file(data_folder+"bbox.txt")
    calib_folder = os.path.join(output_folder, "calib_files")
    kitti_folder = os.path.join(output_folder, "kitti_files")
    
    os.makedirs(calib_folder, exist_ok=True)
    os.makedirs(kitti_folder, exist_ok=True)

    ls(data_folder)

    # Reading files
    frams_mats_e2w, frams_mats_w2e = process_transform_file(data_folder+"infra/pose.txt")
    frams_mats_i2w, frams_mats_w2i = process_transform_file(data_folder+"infra/pose.txt")
    frams_mats_c2w = []
    frams_mats_w2c = []
    for i in range(1, FRAME_COUNT):
        fram_mats_c2w, fram_mats_w2c = process_transform_file(data_folder+"bbox/"+str(i)+".txt")
        frams_mats_c2w.append(fram_mats_c2w)
        frams_mats_w2c.append(fram_mats_w2c)

    total_frames = len(frams_mats_e2w)
    for frame_index in range(total_frames):
        frame_kitti_path = os.path.join(kitti_folder, f"kitti_{frame_index:06}.txt")
        frame_calib_path = os.path.join(calib_folder, f"calib_{frame_index:06}.txt")
        
        extrinsics_available = frame_index < len(frams_mats_w2c) and frams_mats_w2c[frame_index]
        
        with open(frame_kitti_path, 'w') as kitti_file, open(frame_calib_path, 'w') as calib_file:
            # Write calibration data
            if extrinsics_available:
                extrinsics = frams_mats_w2c[frame_index][0]
                K = calculate_camera_intrinsics(90, 1058, 758)
                P = calculate_projection_matrix(K, extrinsics)
                calib_file.write(f"P0: {' '.join(f'{num:.6e}' for num in P.flatten())}\n")
                calib_file.write(f"K: {' '.join(f'{num:.6e}' for num in K.flatten())}\n")

            # Write KITTI format lines 
            if frame_index < len(frams_mats_c2w):
                for bb_index, bb in enumerate(bbs):
                    bb_cw = apply_transform(bb, frams_mats_c2w[frame_index][bb_index])
                    bb_ci = apply_transform(bb_cw, np.linalg.inv(frams_mats_i2w[frame_index]))

                    dimensions = calculate_dimensions(bb_cw)
                    location = calculate_centroid(bb_ci)
                    rotation_y = estimate_yaw(bb_ci)
                    alpha = calculate_alpha(location, rotation_y)

                    kitti_line = format_kitti_bbox("DontCare", bb_ci, dimensions, location, rotation_y, alpha)
                    kitti_file.write(kitti_line)
                    
        print(f"Processed frame {frame_index + 1}/{total_frames}.")

    print(f"Output KITTI data and calibration files written to {output_folder}")

if __name__ == "__main__":
    main()