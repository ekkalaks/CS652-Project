import xml.etree.ElementTree as ET
import os
import shutil
import torch
import cv2

def create_dataset():
    os.mkdir('dataset')
    os.mkdir('dataset/annotation')
    os.mkdir('dataset/image')
    dataset_array = []
    for filename in os.listdir('Annotations'):
        filedir = os.path.join('Annotations', filename)
        tree = ET.parse(filedir)
        myroot = tree.getroot()
        for x in myroot:
            if x.tag == 'object':
                for y in x:
                    if y.tag == 'name':
                        if y.text == 'person':
                            name,entend = filename.split('.')
                            dataset_array.append(name)
                            tree.write("dataset/annotation/{}".format(filename))

    for imagename in os.listdir('JPEGImages'):
        img_file = os.path.join('JPEGImages',imagename)
        name, entend = imagename.split('.')
        if name in dataset_array:
            shutil.copyfile(img_file, f'dataset/image/{imagename}')

def load_model():
    print('load model')
    model = torch.hub.load('ultralytics/yolov5', 'yolov5l', pretrained=True)

def extracted_data():
    for filename in os.listdir('dataset/annotation'):
        filedir = os.path.join('dataset/annotation', filename)
        tree = ET.parse(filedir)
        myroot = tree.getroot()
        for x in myroot:
            if x.tag == 'object':
                for y in x:
                    if y.tag == 'name':
                        if y.text == 'person':
                            name,entend = filename.split('.')
                            dataset_array.append(name)
                            tree.write("dataset/annotation/{}".format(filename))