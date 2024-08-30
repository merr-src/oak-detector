from argparse import Namespace
from pathlib import Path
import cv2
import depthai as dai
import numpy as np
import time
import random
import argparse
import json

from utilfuncs import movdet
from utilfuncs import detector

seed = 42
random.seed(seed)
np.random.seed(seed)
print(dai.__version__)

#--------parse arguments---------
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", help="model path for inference",
                    default='models/16K_260424_n_bs64_150EP_openvino_2022.1_6shave.blob', type=str)
parser.add_argument("-c", "--config", help="config path for inference",
                    default='models/16K_260424_n_bs64_150EP.json', type=str)
parser.add_argument("-s", "--savepath", help="path to save frames with detections",
                    default='detections', type=str)
parser.add_argument("-t1", "--mmax", help="constant movement detecion (frames) to NN switch ON",
                    default=25, type=int) # 30 fps @current bghist, ksize
parser.add_argument("-t2", "--nodetmax", help="frames with no detections to OFF nn pipeline",
                    default=25, type=int) # 12 fps @yolov8 single class detection
parser.add_argument("-t3", "--maxduration", help="minimal total duration (sec) of algorithm cycle",
                    default=100, type=int)
parser.add_argument("-hst", "--bghist", help="memory buffer (frames) for bgr subtractor history",
                    default=100, type=int)
parser.add_argument("-k", "--ksize", help="kernel size to erode nonzero points after bgr subtractor",
                    default=(10, 10), type=str)
parser.add_argument("-o", "--out", help="dir to save output frames w/detections",
                    default='./frames/', type=str)
args = parser.parse_args()
print(args)

#--------params from parsed args--------
nnpath = Path(args.model)
if not nnpath.exists():
    raise ValueError(f"Path {nnpath} does not exist!")

configpath = Path(args.config)
if not configpath.exists():
    raise ValueError(f"Path {configpath} does not exist!")

savepath = Path.cwd() / args.savepath
if not savepath.exists():
    Path(savepath).mkdir()

with configpath.open() as f:
    config = json.load(f)
nnconfig = config.get("nn_config", {})

if "input_size" in nnconfig:
    w, h = tuple(map(int, nnconfig.get("input_size").split('x')))

metadata = nnconfig.get("NN_specific_metadata", {})
classes = metadata.get("classes", {})
coordinates = metadata.get("coordinates", {})
anchors = metadata.get("anchors", {})
anchormasks = metadata.get("anchor_masks", {})
iouthresh = metadata.get("iou_threshold", {})
confthresh = metadata.get("confidence_threshold", {})
nnmappings = config.get("mappings", {})
labels = nnmappings.get("labels", {})

mov_params = {
    'bghist': args.bghist,
    'ksize': eval(args.ksize),
    'mmax': args.mmax,
}

det_params = {
    'nodetmax': args.nodetmax,
    'w': w,
    'h': h,
    'classes': classes,
    'coordinates': coordinates,
    'anchors': anchors,
    'anchormasks': anchormasks,
    'iouthresh': iouthresh,
    'confthresh': confthresh,
    'labels': labels,
    'nnpath': nnpath,
    'savepath': savepath,
}

to_nn = 'stop'
print(to_nn)
duration = 0
maxduration = args.maxduration # in sec
start = time.monotonic()

while duration < maxduration:
    if to_nn == 'stop':
        to_nn = movdet(**mov_params)
        duration = time.monotonic() - start
        print(duration, to_nn)
    else:
        to_nn = detector(**det_params)
        duration = time.monotonic() - start
        print(duration, to_nn, 'to obj detector')

print(f'Stopped when exceed max duration {maxduration} with current duration {duration:.2f}')