'''
    module io

The most important function is `read_skeleton_file`, which reads and parses
a .skeleton file. The output `bodyinfo` has the following data structure:

- `bodyinfo` is a nested list, whereby `bodyinfo[i][j]` contains information
for the `j`th body at video frame `i`.

- Body information is represented as the `Body` class. In particular,
  `body.joints` is a pandas Dataframe of joint information.
'''

import numpy as np
import pandas as pd

class Body:
    '''
    Stores information of a particular body on a single video frame.
    '''
    def __init__(self, metadata, joints):
        self.bodyID = int(metadata[0])
        self.clipedEdges = int(metadata[1])
        self.handLeftConfidence = int(metadata[2])
        self.handLeftState = int(metadata[3])
        self.handRightConfidence = int(metadata[4])
        self.handRightState = int(metadata[5])
        self.isRestricted = int(metadata[6]) # boolean?
        self.leanX = float(metadata[7])
        self.leanY = float(metadata[8])
        self.trackingState = int(metadata[9])
        self.joints = joints
    
    def __str__(self):
        return f"Body {self.bodyID}\n" + \
            f"  clipedEdges: {self.clipedEdges}\n" + \
            f"  handLeftConfidence: {self.handLeftConfidence}\n" + \
            f"  handLeftState: {self.handLeftState}\n" + \
            f"  handRightConfidence: {self.handRightConfidence}\n" + \
            f"  handRightState: {self.handRightState}\n" + \
            f"  isRestricted: {self.isRestricted}\n" + \
            f"  leanX: {self.leanX}\n" + \
            f"  leanY: {self.leanY}\n" + \
            f"  trackingState: {self.trackingState}\n" + \
            f"  number of joints: {len(self.joints)}"

def read_skeleton_file(filename):
    '''
        read_skeleton_file(filename) -> bodyinfo

    Reads an .skeleton file from "NTU RGB+D 3D Action Recognition Dataset". 
    Adapted from the matlab function `read_skeleton_file`.
    '''
    with open(filename, 'r') as f:
        framecount = int(f.readline()) # no of the recorded frames
        bodyinfo = []
        for _ in range(framecount):
            bodyinfo.append(parse_bodies(f))
    return bodyinfo

def parse_bodies(f):
    '''
        parse_bodies(f) -> bodies

    Parse body info for the current frame using the file object `f`.
    '''
    bodies = []
    bodycount = int(f.readline()) # no of observerd skeletons in current frame
    for _ in range(bodycount):
        metadata = f.readline().split(' ')
        joints = parse_joints(f)
        bodies.append(Body(metadata, joints))
    return bodies

def parse_joints(f):
    '''
        parse_joints(f) -> joints

    Parse joint info for the current body using the file object 'f'. Each
    joint is represented as a row in the `joints` dataframe with columns:

    x y z depthX depthY colorX colorY orientationW orientationX orientationY orientationZ trackingState
    '''
    joint_count = int(f.readline())
    x = np.ndarray(joint_count, dtype=float)
    y = np.ndarray(joint_count, dtype=float)
    z = np.ndarray(joint_count, dtype=float)
    depthX = np.ndarray(joint_count, dtype=float)
    depthY = np.ndarray(joint_count, dtype=float)
    colorX = np.ndarray(joint_count, dtype=float)
    colorY = np.ndarray(joint_count, dtype=float)
    orientationW = np.ndarray(joint_count, dtype=float)
    orientationX = np.ndarray(joint_count, dtype=float)
    orientationY = np.ndarray(joint_count, dtype=float)
    orientationZ = np.ndarray(joint_count, dtype=float)
    trackingState = np.ndarray(joint_count, dtype=int)
    for j in range(joint_count):
        data = f.readline().split(' ')
        x[j] = float(data[0])
        y[j] = float(data[1])
        z[j] = float(data[2])
        depthX[j] = float(data[3])
        depthY[j] = float(data[4])
        colorX[j] = float(data[5])
        colorY[j] = float(data[6])
        orientationW[j] = float(data[7])
        orientationX[j] = float(data[8])
        orientationY[j] = float(data[9])
        orientationZ[j] = float(data[10])
        trackingState[j] = int(data[11])
    return pd.DataFrame({
        'x': x, 'y': y, 'z': z, 'depthX': depthX, 'depthY':depthY,
        'colorX': colorX, 'colorY': colorY, 'orientationW': orientationW,
        'orientationX': orientationX, 'orientationY': orientationY,
        'orientationZ': orientationZ, 'trackingState': trackingState
    })
