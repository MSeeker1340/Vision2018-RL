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

def body_to_coordinates(body):
    return np.vstack((body.joints.x, body.joints.y, body.joints.z)).T

def transform_coordinate(coordinates, neck, left_hip, right_hip):
    sf = scaling_factor(neck, left_hip)
    base = coordinate_base(neck, left_hip, right_hip)/sf
    return np.matmul(coordinates-neck, base.T)

def transform_body(body):
    coordinates = body_to_coordinates(body)
    neck = coordinates[2]
    left_hip = coordinates[12]
    right_hip = coordinates[16]
    transformed = transform_coordinate(coordinates, neck, left_hip, right_hip)
    T = transformed.T
    body.joints["vx"] = T[0]
    body.joints["vy"] = T[1]
    body.joints["vz"] = T[2]
    return transformed

## To use: 
## from skeletrack.preprocessing import transform_body
## transform_body(body_info[0][0])
## body_info[0][0].joints will now contain 3 new columns: vx, vy and vz, which store the transformed coordinates