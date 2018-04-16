
# coding: utf-8

# ## Load necessary modules

# In[1]:


# show images inline
#get_ipython().magic(u'matplotlib inline')

# automatically reload modules when they have changed
#get_ipython().magic(u'load_ext autoreload')
#get_ipython().magic(u'autoreload 2')
import sys
# import keras
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

# use this environment flag to change which GPU to use
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# set the modified tf session as backend in keras
keras.backend.tensorflow_backend.set_session(get_session())

classes = ['black_backpack', 'nine_west_bag', 'meixuan_brown_handbag', 'sm_bdrew_grey_handbag', 'wine_red_handbag', 'sm_bclarre_blush_crossbody', 'mk_brown_wrislet', 'black_plain_bag', 'lmk_brown_messenger_bag', 'sm_peach_backpack', 'black_ameligalanti', 'white_bag']

labels_to_names={}
for i, c in enumerate(classes):
	labels_to_names[i] = c

# ## Load RetinaNet model
if 'resnet' in sys.argv[1]:
    from keras_retinanet.models.resnet import resnet_retinanet as retinanet, custom_objects, download_imagenet
elif 'mobilenet' in sys.argv[1]:
    from keras_retinanet.models.mobilenet import mobilenet_retinanet as retinanet, custom_objects, download_imagenet
elif 'vgg' in sys.argv[1]:
    from keras_retinanet.models.vgg import vgg_retinanet as retinanet, custom_objects, download_imagenet
elif 'densenet' in sys.argv[1]:
    from keras_retinanet.models.densenet import densenet_retinanet as retinanet, custom_objects, download_imagenet
else:
    raise NotImplementedError('Backbone \'{}\' not implemented.'.format(sys.argv[1]))

weights = sys.argv[2]

_, _, model = create_models(
    backbone_retinanet=retinanet,
    backbone=sys.argv[1],
    num_classes=12,
    weights=weights,
    multi_gpu=False,
    freeze_backbone=False
)

# print model summary
print(model.summary())

# ## Run detection on example

# In[5]:


# load image

image = read_image_bgr('./examples/test.jpg')

# copy to draw on
draw = image.copy()
draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

# preprocess image for network
image = preprocess_image(image)
image, scale = resize_image(image)

# process image
start = time.time()
_, _, boxes, nms_classification = model.predict_on_batch(np.expand_dims(image, axis=0))
print("processing time: ", time.time() - start)

# compute predicted labels and scores
predicted_labels = np.argmax(nms_classification[0, :, :], axis=1)
scores = nms_classification[0, np.arange(nms_classification.shape[1]), predicted_labels]

# correct for image scale
boxes /= scale

# visualize detections
for idx, (label, score) in enumerate(zip(predicted_labels, scores)):
    if score < 0.5:
        continue
        
    color = label_color(label)

    b = boxes[0, idx, :].astype(int)
    draw_box(draw, b, color=color)
    
    caption = "{} {:.3f}".format(labels_to_names[label], score)
    draw_caption(draw, b, caption)
    
plt.figure(figsize=(15, 15))
plt.axis('off')
plt.imshow(draw)
plt.show()

