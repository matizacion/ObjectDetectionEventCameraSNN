import cv2
import numpy as np
from src.visualize import vis_utils as vis
from glob import glob


def draw_bboxes(img, boxes, labelmap=vis.LABELMAP):
    """
    draw bboxes in the image img
    """
    colors = cv2.applyColorMap(np.arange(0, 255).astype(np.uint8), cv2.COLORMAP_HSV)
    colors = [tuple(*item) for item in colors.tolist()]

    for i in range(boxes.shape[0]):
        pt1 = (int(boxes['x'][i]), int(boxes['y'][i]))
        size = (int(boxes['w'][i]), int(boxes['h'][i]))
        pt2 = (pt1[0] + size[0], pt1[1] + size[1])
        score = boxes['class_confidence'][i]
        class_id = boxes['class_id'][i]
        class_name = labelmap[class_id % len(labelmap)]
        color = colors[class_id * 60 % 255]
        center = ((pt1[0] + pt2[0]) // 2, (pt1[1] + pt2[1]) // 2)
        cv2.rectangle(img, pt1, pt2, color, 1)
        cv2.putText(img, class_name, (center[0], pt2[1] - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color)
        cv2.putText(img, str(score), (center[0], pt1[1] - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color)


if __name__ == '__main__':
    dataset = "Histogram" # or "SAE"
    path_images = glob(f'dataset/{dataset}/packet/*.png')
    path_images.sort()
    path_images = path_images[:1]
    for i, image in enumerate(path_images):

        im = cv2.imread(image)
        if dataset == "Histogram":
            str_name = f'dataset/{dataset}/labels/' + image.split('/')[-1].split('_histogram_')[0] + '_label_' + image.split('_histogram_')[1].split('.png')[0] + '.npy'
            box = np.load(str_name)
        elif dataset == "SAE":
            str_name = f'dataset/{dataset}/labels/' + image.split('/')[-1].split('_SAE_')[0] + '_label_' + image.split('_SAE_')[1].split('.png')[0] + '.npy'
            box = np.load(str_name)
        

        print(image.split('/')[-1])
        # print(box)
        # box[0]['x'] = 205
        # box[1]['w'] = 25
        # idx = [1,2]
        # box = np.delete(box, idx, axis=0)
        # np.save(str_name, box)


        draw_bboxes(im, box)

        
        
        cv2.imshow('out', im)
        cv2.waitKey(0)
        