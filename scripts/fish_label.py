import xml.etree.ElementTree as ET
import pickle as pkl
import numpy as np
import os
from os import listdir, getcwd
from os.path import join
from scipy.misc import imread, imresize, imsave

orig_labels = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']
dataroot = '/media/emredog/research-data/fish/'
outdir = '/media/emredog/research-data/fish/formatted/'
bboxes_file = '/home/emredog/git/fish/bbox/all.pkl'
dataset = 'train'

TRAIN_AMOUNT = 0.8

classes = os.listdir(dataroot + dataset)
bboxes = pkl.load(open(bboxes_file, 'rb'))

last_train = int(np.ceil(TRAIN_AMOUNT * len(bboxes.items())))
input_idx = np.arange(len(bboxes.items()))

np.random.shuffle(input_idx) # now we shuffled all indices for input boxes


def convert(xmin, ymin, w, h, im_w, im_h):
    dw = 1.0/im_w
    dh = 1.0/im_h
    xcenter = xmin + w/2.0
    ycenter = ymin + h/2.0
    x = xcenter*dw
    w = w*dw
    y = ycenter*dh
    h = h*dh
    return (x,y,w,h)

if not os.path.exists(outdir + 'labels/'):
        os.makedirs(outdir + 'labels/')

list_train_file = open('%s/train.txt'%(outdir), 'w')
list_validation_file = open('%s/validation.txt'%(outdir), 'w')
counter = 0
for imkey, annvalue in bboxes.items():
    counter += 1

    label_lowcase = annvalue['label']
    label_upcase = label_lowcase.upper()
    # file path from original location
    filepath = '%s%s/%s/%s.jpg'%(dataroot, dataset, label_upcase, imkey)
    
    # write the image path to appropriate location
    if counter < last_train:
        
        list_train_file.write(filepath)
    else:
        list_validation_file.write(filepath)
    
    # fetch annotations (could be multiple entries)
    annots = annvalue['annotations']  
        
    out_file = open('%s/labels/%s.txt'%(outdir, imkey), 'w')

    # get image size (needed for convert)
    im = imread(filepath)
    im_h, im_w, _ = im.shape
    cls_id = orig_labels.index(label_upcase) # class index
    for ann in annots:
        w, h, x, y = ann
        scaled_bb = convert(x, y, w, h, im_w, im_h)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in scaled_bb]) + '\n')

list_train_file.close()
list_validation_file.close()

# for year, image_set in sets:
#     if not os.path.exists('VOCdevkit/VOC%s/labels/'%(year)):
#         os.makedirs('VOCdevkit/VOC%s/labels/'%(year))
#     image_ids = open('VOCdevkit/VOC%s/ImageSets/Main/%s.txt'%(year, image_set)).read().strip().split()
#     list_file = open('%s_%s.txt'%(year, image_set), 'w')
#     for image_id in image_ids:
#         list_file.write('%s/VOCdevkit/VOC%s/JPEGImages/%s.jpg\n'%(wd, year, image_id))
#         convert_annotation(year, image_id)
#     list_file.close()

