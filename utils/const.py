import numpy as np
INPUT_SHAPE = [224,224,3]
OUTPUT_SHAPE = [56,56,23]
SIGMA = np.array([
    0.026,
    0.025,
    0.025,
    0.035,
    0.035,
    0.079,
    0.079,
    0.072,
    0.072,
    0.062,
    0.062,
    0.107,
    0.107,
    0.087,
    0.087,
    0.089,
    0.089,
    0.035, 
    0.035,
    0.035,
    0.035,
    0.035, 
    0.035
])
NAMES = [
    "Nose", 
    "Left eye", 
    "Right eye", 
    "Left ear",
    "Right ear",
    "Left shoulder",
    "Right shoulder",
    "Left elbow",
    "Right elbow",
    "Left wrist",
    "Right wrist",
    "Left hip",
    "Right hip",
    "Left knee",
    "Right knee",
    "Left ankle",
    "Right ankle",
    "Left big toe",
    "Left small toe",
    "Left heel",
    "Right big toe",
    "Right small toe",
    "Right heel",
]