import glob
import numpy as np

#modified version of the original filter_boxes.py -src.io.box_filtering
def filter_boxes(boxes, skip_ts=int(5e5), min_box_diag=60, min_box_side=20):
    """Filters boxes according to the paper rule. 


    format of boxes:
    ts: timestamp
    x: x position
    y: y position
    w: width
    h: height
    class_id: class

    Returns:
        boxes: filtered boxes
    """
    ts = boxes['ts'] 
    width = boxes['w']
    height = boxes['h']
    diag_square = width**2+height**2
    print(boxes)
    mask = (ts>skip_ts)*(diag_square >= min_box_diag**2)*(width >= min_box_side)*(height >= min_box_side)
    return boxes[mask]


if "__main__" == __name__:
    
    path_boxes = glob.glob('dataset/dataset/*.npy')
    # print(len(path_boxes))
    for path_box in path_boxes:
        boxes = np.load(path_box)
        boxes = filter_boxes(boxes,min_box_diag=30, min_box_side=10)
        np.save(path_box, boxes)
