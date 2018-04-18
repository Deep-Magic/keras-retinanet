# coding: utf-8

import sys
import keras
import keras_retinanet
from keras_retinanet.models.resnet import custom_objects
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color

# import miscellaneous modules
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import time
import argparse

# set tf backend to allow memory to grow, instead of claiming everything
import tensorflow as tf
from keras_retinanet.bin.train import create_models
def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)

def intersects(r1, r2):
    return not (r1[2] < r2[0] or r1[0] > r2[2] or r1[3] < r2[1] or r1[1] > r2[3])

def area(a, b):  # returns None if rectangles don't intersect
    dx = min(a[2], b[2]) - max(a[0], b[0])
    dy = min(a[3], b[3]) - max(a[1], b[1])
    if (dx>=0) and (dy>=0):
        return dx*dy

def union(a,b):
    return (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1])

def return_objects(model_path, class_path, image_path, iou_threshold=0.25, model='resnet152'):

    with open(class_path, 'r') as f:
        classes = [x.strip() for x in f.readlines()]
    
    labels_to_names={}
    for i, c in enumerate(classes):
	    labels_to_names[i] = c
        
    # ## Load RetinaNet model
    if 'resnet' in model_path:
        from keras_retinanet.models.resnet import resnet_retinanet as retinanet, custom_objects, download_imagenet
    elif 'mobilenet' in model_path:
        from keras_retinanet.models.mobilenet import mobilenet_retinanet as retinanet, custom_objects, download_imagenet
    elif 'vgg' in model_path:
        from keras_retinanet.models.vgg import vgg_retinanet as retinanet, custom_objects, download_imagenet
    elif 'densenet' in model_path:
        from keras_retinanet.models.densenet import densenet_retinanet as retinanet, custom_objects, download_imagenet
    else:
        raise NotImplementedError('Backbone \'{}\' not implemented.'.format(model_path))

    _, _, model = create_models(
        backbone_retinanet=retinanet,
        backbone=model,
        num_classes=len(classes),
        weights=model_path,
        multi_gpu=False,
        freeze_backbone=False
    )

    image = read_image_bgr(image_path)

    # preprocess image for network
    image = preprocess_image(image)
    image, scale = resize_image(image)

    # process image
    start = time.time()
    _, _, boxes, nms_classification = model.predict_on_batch(np.expand_dims(image, axis=0))
    #print("processing time: ", time.time() - start)

    # compute predicted labels and scores
    predicted_labels = np.argmax(nms_classification[0, :, :], axis=1)
    scores = nms_classification[0, np.arange(nms_classification.shape[1]), predicted_labels]

    # correct for image scale
    boxes /= scale
    
    plabels, pscores, pboxes = [], [], []
    for idx, (label, score) in enumerate(zip(predicted_labels, scores)):
        if score<0.5:
            continue
        pboxes.append(boxes[0, idx, :].astype(int))
        plabels.append([label])
        pscores.append(score)
    
    rboxes, rscores, rlabels, ignore, inter = [], [], [], [], False
    for i in range(len(pboxes)):
        if i in ignore:
            continue
        for j in range(i+1, len(pboxes)):
            if intersects(pboxes[i], pboxes[j]):
                iou = area(pboxes[i], pboxes[j])/float(union(pboxes[i], pboxes[j])-area(pboxes[i], pboxes[j]))
                if pscores[i]<pscores[j] and iou>iou_threshold:
                    inter = True
                elif pscores[i]>pscores[j] and iou>iou_threshold:
                    ignore.append(j)
                    plabels[i].append(plabels[j][0])
        
        if not inter:                
            rboxes.append(pboxes[i])
            rscores.append(pscores[i])
            rlabels.append(plabels[i])
        else:
            inter = False
        
    return rlabels, rscores, rboxes, labels_to_names

if __name__=='__main__':
    # use this environment flag to change which GPU to use
    #os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--image', help='Path to demo image file', required=True)
    ap.add_argument('-m', '--model', help='Path to model', required=True)
    ap.add_argument('-c', '--classf', help='Path to classfile', required=True)
    ap.add_argument('-r', '--iou', help='IOU Threshold', required=True)
    args = ap.parse_args()
    # set the modified tf session as backend in keras
    keras.backend.tensorflow_backend.set_session(get_session())
    
    demo_img_path = args.image
    iou_threshold = float(args.iou)
    
    draw = read_image_bgr(demo_img_path)
    draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)
    
    predicted_labels, scores, boxes, labels_to_names = return_objects(args.model, args.classf, demo_img_path, iou_threshold)   
        
    # visualize detections
    for idx, (labels, score) in enumerate(zip(predicted_labels, scores)):
        
        color = label_color(labels[0])

        b = boxes[idx].astype(int)
        draw_box(draw, b, color=color)
        
        caption = "{} {:.3f}".format(labels_to_names[labels[0]], score)
        if len(labels)>1:
            for lid in labels[1:]:
                caption+='\n'+labels_to_names[lid]
            
        draw_caption(draw, b, caption)
        
    plt.figure(figsize=(15, 15))
    plt.axis('off')
    plt.imshow(draw)
    plt.show()

