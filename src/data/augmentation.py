
import numpy as np
import random

def normalize(points):
    mean = np.mean(points, axis=0)
    centered_points = points - mean
    scale = (1 / np.abs(centered_points).max()) * 0.9999999
    normalized_points = centered_points * scale
    return normalized_points

def random_rotation(point_cloud):
    # Calculate the centroid of the point cloud
    centroid = np.mean(point_cloud, axis=0, keepdims=True)

    # Center the point cloud at the origin
    centered_point_cloud = point_cloud - centroid

    # Generate random rotation angles
    angles = np.random.uniform(0, 2*np.pi, size=3)
    cos_vals = np.cos(angles)
    sin_vals = np.sin(angles)

    # Rotation matrices for x, y, z
    Rx = np.array([[1, 0, 0],
                   [0, cos_vals[0], -sin_vals[0]],
                   [0, sin_vals[0], cos_vals[0]]])

    Ry = np.array([[cos_vals[1], 0, sin_vals[1]],
                   [0, 1, 0],
                   [-sin_vals[1], 0, cos_vals[1]]])

    Rz = np.array([[cos_vals[2], -sin_vals[2], 0],
                   [sin_vals[2], cos_vals[2], 0],
                   [0, 0, 1]])

    # Combined rotation matrix
    R = np.dot(np.dot(Rz, Ry), Rx)

    # Rotate the point cloud
    rotated_point_cloud = np.dot(centered_point_cloud, R.T)

    # Move the point cloud back to its original centroid
    rotated_point_cloud += centroid
    return rotated_point_cloud

def jitter(point_cloud, sigma=0.01, clip=0.05):
    jittered_data = np.clip(sigma * np.random.randn(*point_cloud.shape), -clip, clip)
    jittered_point_cloud = point_cloud + jittered_data
    return jittered_point_cloud


def random_transform(point_cloud):
    transforms = [random_rotation, jitter]
    transform = random.choice(transforms)
    return transform(point_cloud)

