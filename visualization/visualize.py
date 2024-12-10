# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

"""
small executable to show the content of the Prophesee dataset
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import numpy as np
import cv2
import argparse
from glob import glob
import random
import yaml

from src.visualize import vis_utils as vis

from src.io.psee_loader import PSEELoader


def play_files_parallel(td_files, labels=None, delta_t=50000, skip=0):
    """
    Plays simultaneously files and their boxes in a rectangular format.
    """
    # open the video object for the input files
    videos = [PSEELoader(td_file) for td_file in td_files]
    # use the naming pattern to find the corresponding box file
    box_videos = [PSEELoader(glob(td_file.split('_td.dat')[0] +  '*.npy')[0]) for td_file in td_files]

    height, width = videos[0].get_size()
    labelmap = vis.LABELMAP if height == 240 else vis.LABELMAP_LARGE

    # optionally skip n minutes in all videos
    for v in videos + box_videos:
        v.seek_time(skip)

    # preallocate a grid to display the images
    size_x = int(math.ceil(math.sqrt(len(videos))))
    size_y = int(math.ceil(len(videos) / size_x))
    frame = np.zeros((size_y * height, width * size_x, 3), dtype=np.uint8)

    cv2.namedWindow('out', cv2.WINDOW_NORMAL)

    # while all videos have something to read
    while not sum([video.done for video in videos]):

        # load events and boxes from all files
        events = [video.load_delta_t(delta_t) for video in videos]
        box_events = [box_video.load_delta_t(delta_t) for box_video in box_videos]
        # print(box_events)
        for index, (evs, boxes) in enumerate(zip(events, box_events)):
            y, x = divmod(index, size_x)
            # put the visualization at the right spot in the grid
            im = frame[y * height:(y + 1) * height, x * width: (x + 1) * width]
            # call the visualization functions
            im = vis.make_binary_histo(evs, img=im, width=width, height=height)

            vis.draw_bboxes(im, boxes, labelmap=labelmap)

        # display the result
        cv2.imshow('out', frame)
        cv2.waitKey(0)

def visualizeGEN1Format(file_dataset_files):
    play_files_parallel([file_dataset_files[10]], skip=0, delta_t=100000)
    print(file_dataset_files[0])

def visualizeYoloFormat(yaml_path, image_path, labels_path):

    with open(yaml_path, 'r') as stream:
        try:
            data = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)    
    # print(data)
    colors = {0:(0,0,255), 1 : (0,255,0)}
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    image_paths = glob(image_path + "/*.png")
    labels_paths = glob(labels_path + "/*.txt")
    for image_path in image_paths:
        img = cv2.imread(image_path)
        image_name = image_path.split("/")[-1]
        label_name = image_name.replace(".png", ".txt")
        label_name = label_name.replace("histogram", "label")
        corresponding_label = labels_path + "/" + label_name
        with open(corresponding_label, 'r') as file:
            lines = file.readlines()
            for line in lines:
                line = line.split()
                # print(line)
                x = int(float(line[1]) * img.shape[1])
                y = int(float(line[2]) * img.shape[0])
                w = int(float(line[3]) * img.shape[1])
                h = int(float(line[4]) * img.shape[0])
                lTop = (int(x-w/2), int(y-h/2))
                rBottom = (int(x+w/2), int(y+h/2))
                # print(colors[int(line[0])])
                cv2.rectangle(img, lTop, rBottom, colors[int(line[0])], 1)
                cv2.putText(img,data['names'][int(line[0])], (lTop[0], lTop[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[int(line[0])], 1)
        
        cv2.imshow('image', img)
        cv2.waitKey(0)
    

if __name__ == '__main__':
    # file_dataset_files = glob('dataset/dataset/*.dat')
    # file_dataset_files = sorted(file_dataset_files)
    # visualizeGEN1Format(file_dataset_files)

    dataset_path = "dataset/GEN1"
    yaml_path = dataset_path + "/data.yaml"
    image_path = dataset_path + "/done"
    labels_path = dataset_path + "/NonNegativeLabelsYolo"
    visualizeYoloFormat(yaml_path, image_path, labels_path)
