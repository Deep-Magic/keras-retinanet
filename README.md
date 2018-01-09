This repo contains the trained models and code for the application of a RetinaNet model over the Bags dataset given by the company. 

Location of trained model checkpoints:

ls keras-retinanet/keras_retinanet/snapshots/
resnet101_csv_orig_dataset_10.h5  resnet50_csv_dataset1_12.h5
resnet101_csv_orig_dataset_11.h5  resnet50_csv_orig_dataset_13.h5
resnet101_csv_orig_dataset_12.h5  resnet50_csv_orig_dataset_14.h5
resnet50_csv_dataset1_10.h5       resnet50_csv_orig_dataset_15.h5
resnet50_csv_dataset1_11.h5

I've currently trained a few models with ResNet 50 and ResNet 101 backbone. I've also tried to crawl through the web to generate more similar bags images. orig_dataset refers to the Bags dataset provided by the company. Dataset1 is a mix of both company dataset and the crawled images. 

I've found that the ResNet 101 model's latest checkpoint file `resnet101_csv_orig_dataset_12.h5` gives the best performance in detection and classification of all the models I've trained so far. 

In order to run the demo of the model over a video of your choice, use:

python examples/demo.py keras_retinanet/snapshots/resnet101_csv_orig_dataset_12.h5 <PATH_TO_VIDEO_FILE>
 
