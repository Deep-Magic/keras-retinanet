import csv
import glob
import os
from colour_segmentor import find_bbox
import xml.etree.ElementTree as ET

with open('annotations_full.csv', 'w') as csvfile:

	filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)

	# Create CSV dataset from Pascal VOC format dataset
	# The dataset (Pascal VOC format) is located in keras-retinanet/keras_retinanet/Data/handbag_images/

	for f in glob.glob(os.getcwd()+'/Data/handbag_images/JPEGImages/*.png'):
	    #print (f)
	    parts = f.split('/')
	    annot = '/'.join(parts[:-2])+'/Annotations/'+parts[-1][:-3]+'xml'
	    tree = ET.parse(annot)
	    root = tree.getroot()         

	    for obj in root.findall('object'):
	   
		cls = obj.find('name').text
		bx = [int(obj.find('bndbox').find('xmin').text), int(obj.find('bndbox').find('ymin').text), int(obj.find('bndbox').find('xmax').text), int(obj.find('bndbox').find('ymax').text)]
		filewriter.writerow([f, str(bx[0]), str(bx[1]), str(bx[2]), str(bx[3]), cls])   

	'''
	# Create CSV dataset from regular images and use colour contrast to find segmentation masks / bounding boxes
	# Located in keras-retinanet/keras_retinanet/Data/bags2/
	# Format: (bags/<class_folder>*/images*)*

	for f in glob.glob(os.getcwd()+'/Data/bags2/*'):
	    for g in glob.glob(f+'/*.jpg'):
		bbox = find_bbox(g)
		filewriter.writerow([g, str(bbox[0]), str(bbox[1]), str(bbox[2]), str(bbox[3]), f.split('/')[-1]])   
	'''

#Classes for initial bags dataset

classes = ['black_backpack', 'nine_west_bag', 'meixuan_brown_handbag', 'sm_bdrew_grey_handbag', 'wine_red_handbag', 'sm_bclarre_blush_crossbody', 'mk_brown_wrislet', 'black_plain_bag', 'lmk_brown_messenger_bag', 'sm_peach_backpack', 'black_ameligalanti', 'white_bag']   

with open('classes_full.csv', 'w') as csvfile:
	filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
	for i, c in enumerate(classes):
	    filewriter.writerow([c, str(i)])
