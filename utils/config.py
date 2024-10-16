import numpy as np

BASE_DIR_WITH_ANNOTATIONS = './data/annotated_videos'

HSV_map = {
    'kam4': {
        'HSV_red_lower': np.array([0, 170, 40]),
        'HSV_red_upper': np.array([11, 255, 255]),
        'HSV_blue_lower': np.array([105, 97, 0]),
        'HSV_blue_upper': np.array([129, 255, 182])
    },
    'kam2': {
        'HSV_red_lower': np.array([0, 170, 30]),
        'HSV_red_upper': np.array([10, 255, 255]),
        'HSV_blue_lower': np.array([100, 0, 0]),
        'HSV_blue_upper': np.array([179, 255, 103])
    }
}