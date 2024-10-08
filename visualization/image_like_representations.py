import math
import numpy as np
import cv2
import argparse
from glob import glob
import random

from src.visualize import vis_utils as vis

from src.io.psee_loader import PSEELoader


def SAE(events, width=304, height=240, current_time = 0, delta_t=100000):
    """
    Surface of Active Events
    """
    img = np.ones((height, width, 3))*current_time
    img[events['y'], events['x'], :] = events['t'].reshape(-1, 1)
    img  = np.floor(255*(img - current_time)/delta_t).astype(np.uint8)
    return img


def representations2D(td_file, delta_t=100000, label=False, representation=['histogram']):

    # representation should be one of elements in ['histogram', 'SAE']

    video = PSEELoader(td_file)
    height, width = video.get_size()
    if label:
        box_video = PSEELoader(glob(td_file.split('_td.dat')[0] +  '*.npy')[0])
        labelmap = vis.LABELMAP if height == 240 else vis.LABELMAP_LARGE

    cv2.namedWindow('out', cv2.WINDOW_NORMAL)

    counter = 0

    if len(representation) == 1:
        representation = representation[0]
        while not (video.done):
            events = video.load_delta_t(delta_t)
            boxes = box_video.load_delta_t(delta_t)
            if boxes.size == 0:
                continue
            current_time = video.current_time
            if representation == 'histogram':
                im = vis.make_binary_histo(events, width=width, height=height)
            elif representation == 'SAE':
                im = SAE(events, width=width, height=height, current_time = current_time, delta_t=delta_t)
            elif representation == 'both':
                im = np.zeros((height, width, 3), dtype=np.uint8)
                im[events['y'], events['x'], :] = 255 * events['p'][:, None]
            if label:
                vis.draw_bboxes(im, boxes, labelmap=labelmap)
            cv2.imshow('out', im)
            cv2.waitKey(1)
            counter += 1
    else:
        size_x = len(representation)
        frame = np.zeros((height, width * size_x, 3), dtype=np.uint8)
        index_pic = 0
        while not (video.done):
            events = video.load_delta_t(delta_t)
            boxes = box_video.load_delta_t(delta_t)
            if boxes.size == 0:
                continue
            current_time = video.current_time
            list_im = [np.zeros((height, width, 3), dtype=np.uint8) for _ in range(len(representation))]
            for rep in representation:
                if rep == 'histogram':
                    list_im[representation.index(rep)] = vis.make_binary_histo(events, width=width, height=height)
                    path_to_save = 'dataset/Histogram/img/' + td_file.split('/')[-1].split('_td.dat')[0] + '_histogram_' + str(index_pic) + '.png'
                    cv2.imwrite(path_to_save, list_im[representation.index(rep)])
                elif rep == 'SAE':
                    list_im[representation.index(rep)] = SAE(events, width=width, height=height, current_time = current_time, delta_t=delta_t)
                    path_to_save = 'dataset/SAE/img/' + td_file.split('/')[-1].split('_td.dat')[0] + '_SAE_' + str(index_pic) + '.png'
                    cv2.imwrite(path_to_save, list_im[representation.index(rep)])
                
            if label:
                for im in list_im:
                    vis.draw_bboxes(im, boxes, labelmap=labelmap)
                    np.save('dataset/SAE/labels/' + td_file.split('/')[-1].split('_td.dat')[0] + '_label_' + str(index_pic) + '.npy', boxes)
                    np.save('dataset/Histogram/labels/' + td_file.split('/')[-1].split('_td.dat')[0] + '_label_' + str(index_pic) + '.npy', boxes)
            for index, im in enumerate(list_im):
                y, x = divmod(index, size_x)
                frame[y * height:(y + 1) * height, x * width: (x + 1) * width] = im
            cv2.resizeWindow('out', 800, 400)
            cv2.imshow('out', frame)
            cv2.waitKey(1)
            counter += 1
            index_pic += 1
    return counter

if __name__ == '__main__':
    file_dataset_files = glob('dataset/dataset/*.dat')
    file_dataset_files = sorted(file_dataset_files)
    counter = 0
    # for file_dataset in file_dataset_files:
    #     # print(file_dataset)
    #     counter += representations2D(file_dataset, label=True, representation=['histogram', 'SAE'], delta_t=100000)
    # print(counter)
