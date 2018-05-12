import glob
import argparse
import os
import numpy as np
import json
import csv

from PIL import Image

if __name__=='__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument("-a", "--annotation_file", required=True, help="Path to CSV file")
    ap.add_argument("-c", "--class_file", required=True, help="Path of CSV file with output classes")
    ap.add_argument("-o", "--output_json", required=False, help="Path to output JSON file", default="csv_annotations")
    args = vars(ap.parse_args())

    with open(args['class_file'], 'r') as f:
        classes = sorted([x.strip().split(',')[0] for x in f.readlines()])
    
    images, anns = [], []
    
    with open(args['annotation_file'], 'r') as f:
        textdata = [x.strip().split(',') for x in f.readlines()] 
    
    img_files = set()
    [img_files.add(x[0]) for x in textdata]
    img_files = sorted(list(img_files))
    
    for i, f in enumerate(img_files):
        img = Image.open(os.path.join(os.path.dirname(args['annotation_file']), f))
        width, height = img.size
        dic = {'file_name': os.path.join(os.path.dirname(args['annotation_file']), f), 'id': i, 'height': height, 'width': width}
        images.append(dic)
    
    ann_index = 0
    for i, x in enumerate(textdata):
        img_id = img_files.index(x[0])
        bbox = [int(p) for p in x[1:-1]]
        cls = classes.index(x[-1])+1
        xmin, ymin, xmax, ymax = bbox
        poly = [[xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax, xmin, ymin]]
        area = (xmax-xmin)*(ymax-ymin)
        
        dic2 = {'segmentation': poly, 'area': area, 'iscrowd':0, 'image_id':img_id, 'bbox':bbox, 'category_id': cls, 'id': i}
        anns.append(dic2)

    data = {'images':images, 'annotations':anns, 'categories':[], 'classes': classes}

    with open(args['output_json']+'.json', 'w') as outfile:
        json.dump(data, outfile)
