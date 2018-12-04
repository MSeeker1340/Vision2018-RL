import numpy as np

def coordinate_base(neck, left_hip, right_hip):
    u1 = (left_hip - neck)
    u1 /= np.linalg.norm(u1)
    u2 = (right_hip - neck)
    u3 = np.cross(u1, u2)/np.linalg.norm(u2)
    u2 = np.cross(u1, u3)
    return np.vstack((u1, u2, u3))

def scaling_factor(neck, left_hip):
    return np.linalg.norm(left_hip - neck)

def transform_coordinate(coordinates, base, sf):
    base2 = base/sf
    return np.matmul(coordinates, base2.T)