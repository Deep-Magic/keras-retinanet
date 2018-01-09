# coding: utf-8
from __future__ import print_function

import keras
import keras.preprocessing.image
from keras_retinanet.models.resnet import custom_objects
from keras_retinanet.preprocessing.csv_generator import CSVGenerator
from keras_retinanet.utils.image import preprocess_image

import matplotlib.pyplot as plt
import cv2
import os
import math
import numpy as np
import csv
import time
import sys

import tensorflow as tf

def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)

classes = ['black_backpack', 'nine_west_bag', 'meixuan_brown_handbag', 'sm_bdrew_grey_handbag', 'wine_red_handbag', 'sm_bclarre_blush_crossbody', 'mk_brown_wrislet', 'black_plain_bag', 'lmk_brown_messenger_bag', 'sm_peach_backpack', 'black_ameligalanti', 'white_bag']

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
keras.backend.tensorflow_backend.set_session(get_session())

model = keras.models.load_model(sys.argv[1], custom_objects = custom_objects)
print ('\n\n............................ LOADING COMPLETE ........................\n\n')
print ('RetinaNet model', sys.argv[1], 'loaded successfully!')

cap = cv2.VideoCapture(sys.argv[2])

cv2.namedWindow('Detection Results')

vid_count = int(cap.get(7))

if not os.path.exists('video_frames'):
    os.makedirs('video_frames')
i = 0

files = [name for name in os.listdir('video_frames/')]
if (len(files)!=vid_count):
    print ('Deleting previously cached video frames and annotations if any!') 
    for f in files:
        os.remove('video_frames/'+f)
    if (os.path.exists('annotations_video.csv')):
        os.remove('annotations_video.csv')
    if (os.path.exists('classes_video.csv')):
        os.remove('classes_video.csv')    
else:
    print ('Using cached image frames for speedup!')
#print (vid_count, vid_height, vid_width, vid_fps)

if (not os.path.exists('annotations_video.csv')): 
    with open('annotations_video.csv', 'w') as csvfile:

        filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        while(1):
            ret, image = cap.read()
            if (not ret):
                break
        
            if (not os.path.exists('video_frames/'+str(i)+'.jpg')):
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                cv2.imwrite('video_frames/'+str(i)+'.jpg', image)
            filewriter.writerow([os.getcwd()+'/video_frames/'+str(i)+'.jpg', '', '', '', '', ''])      
            i+=1

if (not os.path.exists('classes_video.csv')):
    with open('classes_video.csv', 'w') as csvfile:

        filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)

        for i, c in enumerate(classes):
            filewriter.writerow([c, str(i)]) 

#print(model.summary())

#create image data generator object
val_image_data_generator = keras.preprocessing.image.ImageDataGenerator()

# create a generator for testing data
val_generator = CSVGenerator(
                'annotations_video.csv',
                'classes_video.csv',
                val_image_data_generator,
                batch_size=1
            )

print ('Running Detection on the frames from video!')

for index in range(val_generator.size()):

    # load image
    image = val_generator.load_image(index)

    # copy to draw on
    draw = image.copy()
    draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

    # preprocess image for networka
    image = val_generator.preprocess_image(image)
    image, scale = val_generator.resize_image(image)
    annotations = val_generator.load_annotations(index)

    # process image
    start = time.time()
    _, _, detections = model.predict_on_batch(np.expand_dims(image, axis=0))
    print("Frame", index, '/', val_generator.size(),  "Time Taken: ", time.time() - start, 'seconds')

    # compute predicted labels and scores
    predicted_labels = np.argmax(detections[0, :, 4:], axis=1)
    scores = detections[0, np.arange(detections.shape[1]), 4 + predicted_labels]

    # correct for image scale
    detections[0, :, :4] /= scale

    # visualize detections
    for idx, (label, score) in enumerate(zip(predicted_labels, scores)):
        if score < 0.5:
            continue
        b = detections[0, idx, :4].astype(int)
        
        cv2.rectangle(draw, (b[0], b[1]), (b[2], b[3]), (255, 20, 20), 2)
        
        caption = "{}".format(val_generator.label_to_name(label))
        #print("{} ({}%)".format(val_generator.label_to_name(label), math.ceil(score*100)))
        #cv2.putText(draw, caption, (b[0]-15, b[1] - 15), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
        
        h, w = cv2.getTextSize(caption, cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, 1)[0]
        if ((b[0]+h)>draw.shape[1]): #ensure text stays inside image
            b[0] = b[0] - (b[0]+h-draw.shape[1])
            h = h - (b[0]+h-draw.shape[1])
            b[1] = b[1] +20
            
        
        cv2.rectangle(draw,((b[0], b[1])),(b[0]+h, b[1]+w+10),(0, 0, 0), -1)
        cv2.putText(draw, caption, (b[0], b[1]+10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (255,255,255), 1)
        #cv2.putText(draw, caption, (int(b[0]-10), int((b[1]+b[3])*0.5)), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 0), 1)
        #draw = cv2.copyMakeBorder(draw, 10,10, 10, 10, cv2.BORDER_CONSTANT, (255,255,255))

    '''       
    # visualize annotations
    for annotation in annotations:
        label = int(annotation[4])
        b = annotation[:4].astype(int)
        print(val_generator.label_to_name(label))
        cv2.rectangle(draw, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 2)
        caption = "{}".format(val_generator.label_to_name(label))
        cv2.putText(draw, caption, (b[0], b[1] - 10), 5, 1.5, (0, 0, 0), 3)
        cv2.putText(draw, caption, (b[0], b[1] - 10), 5, 1.5, (255, 255, 255), 2)
        
    plt.figure(figsize=(10, 10))
    plt.axis('off')
    plt.imshow(draw)
    #print(draw)
    plt.show()
    '''
    #draw = cv2.copyMakeBorder(draw, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=(255,255,255))
    cv2.imshow('Detection Results', draw)
    
    cv2.waitKey(1)

cv2.destroyAllWindows()
