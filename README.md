This repo contains the trained models and code for the application of a RetinaNet model described in the paper [Focal Loss for Dense Object Detection](https://arxiv.org/pdf/1708.02002.pdf) over the Bags dataset. 

The code base is from [Keras-RetinaNet](https://github.com/fizyr/keras-retinanet) for the Keras implementation of RetinaNet.

### INSTALLATION:

1. Clone the repo.

```shell
cd keras-retinanet/
sudo python2 setup.py install --user
```

2. Pre-requisites (Keras and Keras-ResNet):

```shell
sudo pip2 install keras
sudo pip2 install --user --upgrade git+https://github.com/broadinstitute/keras-resnet
```

3. Download the trained checkpoint file: ([Link to ResNet-101 checkpoint](https://drive.google.com/file/d/1SIVXQ6qP4eJo2tXV90xUCqYGS927Dok6/view?usp=sharing)) and place it in `keras-retinanet/keras_retinanet/snapshots/`

I've currently trained a few models with ResNet 50 and ResNet 101 backbone. I've also tried to crawl through the web to generate more similar bags images. I used the Bags dataset provided by the company. I also tried to crawl through shopping portals for handbag images that I could augment with the original dataset but it skewed with the results.  

I've found that the ResNet 101 model's latest checkpoint file `resnet101_csv_orig_dataset_12.h5` gives the best performance in detection and classification of all the models I've trained so far. 

### RUNNING DEMO:

In order to run the demo of the model over a video of your choice, use:

```shell

cd keras-retinanet/
python examples/demo.py keras_retinanet/snapshots/resnet101_csv_orig_dataset_12.h5 <PATH_TO_VIDEO_FILE>
``` 
