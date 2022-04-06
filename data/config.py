# config.py
import os.path

# gets home dir cross platform
HOME = 'data'

# for making bounding boxes pretty
COLORS = ((255, 0, 0, 128), (0, 255, 0, 128), (0, 0, 255, 128),
          (0, 255, 255, 128), (255, 0, 255, 128), (255, 255, 0, 128))

MEANS = (104, 117, 123)

# DAGNet CONFIGS
voc_dagnet = {
    '512': {
        'num_classes': 2,  # pre-defined classes + background class
        'lr_steps': (5000, 15000, 80000, 120000),
        'max_iter': 120000,
        'feature_maps': [64*2, 32*2, 16*2, 8*2],
        'min_dim': 512,
        'steps': [8/2, 16/2, 32/2, 64/2],
        'min_sizes': [16, 32, 64, 128],
        'max_sizes': [],
        'aspect_ratios': [[2], [2], [2], [2]],
        'variance': [0.1, 0.2],
        'clip': True,
        'name': 'DAGNet_VOC_320',
    }
}
