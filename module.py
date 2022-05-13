import xml.etree.ElementTree as ET
import os
import torch
import cv2
from pixellib.torchbackend.instance import instanceSegmentation
import matplotlib.pyplot as plt

# def create_dataset():
#     print('create person dataset')
#     os.mkdir('dataset')
#     os.mkdir('dataset/annotation')
#     os.mkdir('dataset/image')
#     dataset_array = []
#     for filename in os.listdir('Annotations'):
#         filedir = os.path.join('Annotations', filename)
#         tree = ET.parse(filedir)
#         myroot = tree.getroot()
#         for x in myroot:
#             if x.tag == 'object':
#                 for y in x:
#                     if y.tag == 'name':
#                         if y.text == 'person':
#                             name,entend = filename.split('.')
#                             dataset_array.append(name)
#                             tree.write("dataset/annotation/{}".format(filename))
#
#     for imagename in os.listdir('JPEGImages'):
#         img_file = os.path.join('JPEGImages',imagename)
#         name, entend = imagename.split('.')
#         if name in dataset_array:
#             shutil.copyfile(img_file, f'dataset/image/{imagename}')

def load_model():
    print('load model')
    model = torch.hub.load('ultralytics/yolov5', 'yolov5l', pretrained=True)

    segment_image = instanceSegmentation()
    segment_image.load_model("pointrend_resnet101.pkl", network_backbone="resnet101")
    target_classes = segment_image.select_target_classes(person=True)
    print('load successfully')

    return model,segment_image,target_classes

def extracted_data():
    print('load bounding box')
    bnd_person_array = []
    bnd_no_person_array = []
    for filename in os.listdir('Annotations'):
        filedir = os.path.join('Annotations', filename)
        tree = ET.parse(filedir)
        myroot = tree.getroot()
        for x in myroot:
            if x.tag == 'object':
                for y in x:
                    if y.tag == 'name':
                        if y.text == 'person':
                            check = True
                        else:
                            check = False

                    if y.tag == 'bndbox':
                        if check:
                            for bnd in y:
                                if bnd.tag == 'xmax':
                                    xmax = bnd.text
                                elif bnd.tag == 'xmin':
                                    xmin = bnd.text
                                elif bnd.tag == 'ymax':
                                    ymax = bnd.text
                                elif bnd.tag == 'ymin':
                                    ymin = bnd.text
                            bnd_person_out = [xmin, ymin, xmax, ymax]
                            name, entend = filename.split('.')
                            bnd_person_array.append([name,bnd_person_out])
                        else:
                            for bnd in y:
                                if bnd.tag == 'xmax':
                                    xmax = bnd.text
                                elif bnd.tag == 'xmin':
                                    xmin = bnd.text
                                elif bnd.tag == 'ymax':
                                    ymax = bnd.text
                                elif bnd.tag == 'ymin':
                                    ymin = bnd.text
                            bnd_no_person_out = [xmin, ymin, xmax, ymax]
                            name, entend = filename.split('.')
                            bnd_no_person_array.append([name,bnd_no_person_out])

    result_out = [bnd_person_array,bnd_no_person_array]
    return result_out

def use_model(frame,model,segment_image,target_classes):
    results = model(frame, size=640)
    out2 = results.pandas().xyxy[0]

    result, image_result = segment_image.segmentFrame(frame, show_bboxes=True,
                                                      segment_target_classes=target_classes,
                                                      extract_segmented_objects=False,
                                                      save_extracted_objects=False)
    return out2, result


def person_detect():
    fall_count_person = 0
    fall_count_noperson = 0
    model, segment_image, target_classes = load_model()
    result_out = extracted_data()

    TP_pixellib = 0
    FP_pixellib = 0
    TN_pixellib = 0
    FN_pixellib = 0
    TP_yolov5 = 0
    FP_yolov5 = 0
    TN_yolov5 = 0
    FN_yolov5 = 0

    for bnd in result_out[0]:
        print(bnd)
        xmin,ymin,xmax,ymax = bnd[1]
        xmin, ymin, xmax, ymax = float(xmin), float(ymin), float(xmax), float(ymax)
        xmin, ymin, xmax, ymax = int(xmin),int(ymin),int(xmax),int(ymax)
        img = cv2.imread(f'JPEGImages/{bnd[0]}.jpg')
        img = img[ymin:ymax,xmin:xmax]

        try:
            output_yolov5,output_pixellib = use_model(img,model, segment_image, target_classes)

            check_yolo = 0
            for i in range(len(output_yolov5)):
                j = i+1
                obj_name = output_yolov5.iat[i, 6]
                if obj_name == 'person':
                    check_yolo += 1
                    if check_yolo == 1:
                        TP_yolov5 += 1
                if j == len(output_yolov5):
                    if check_yolo == 0:
                        FN_yolov5 += 1

            if len(output_pixellib['object_counts']) != 0:
                TP_pixellib += 1
            elif len(output_pixellib['object_counts']) == 0:
                FN_pixellib += 1
        except:
            fall_count_person+=1

    for bnd in result_out[1]:
        print(bnd)
        xmin,ymin,xmax,ymax = bnd[1]
        xmin, ymin, xmax, ymax = float(xmin), float(ymin), float(xmax), float(ymax)
        xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
        img = cv2.imread(f'JPEGImages/{bnd[0]}.jpg')
        img = img[ymin:ymax,xmin:xmax]
        try:
            output_yolov5,output_pixellib = use_model(img,model, segment_image, target_classes)

            check_yolo = 0
            for i in range(len(output_yolov5)):
                j = i+1
                obj_name = output_yolov5.iat[i, 6]
                if obj_name == 'person':
                    check_yolo += 1
                    if check_yolo == 1:
                        FP_yolov5 += 1
                if j == len(output_yolov5):
                    if check_yolo == 0:
                        TN_yolov5 += 1

            if len(output_pixellib['object_counts']) != 0:
                FP_pixellib += 1
            elif len(output_pixellib['object_counts']) == 0:
                TN_pixellib += 1
        except:
            fall_count_noperson+=1

    count_yolov5 = TP_yolov5+FP_yolov5+TN_yolov5+FN_yolov5

    count_pixellib = TP_pixellib+FP_pixellib+TN_pixellib+FN_pixellib
    print(len(result_out[0]), len(result_out[1]))
    print(len(result_out[0]) + len(result_out[1]))
    print(count_yolov5,count_pixellib)
    print(fall_count_person,fall_count_noperson)

    print('---------------------')
    print(TP_yolov5,FP_yolov5,TN_yolov5,FN_yolov5)
    print(TP_pixellib,FP_pixellib,TN_pixellib,FN_pixellib)

    acc_yolov5 = (TP_yolov5+TN_yolov5)/(TP_yolov5+TN_yolov5+FP_yolov5+FN_yolov5)
    recall_yolov5 = (TP_yolov5)/(TP_yolov5+FN_yolov5)
    precision_yolov5 = (TP_yolov5)/(TP_yolov5+FP_yolov5)
    f1_yolov5 = (2*precision_yolov5*recall_yolov5)/(precision_yolov5+recall_yolov5)

    acc_pixellib = (TP_pixellib+TN_pixellib)/(TP_pixellib+TN_pixellib+FP_pixellib+FN_pixellib)
    recall_pixellib = (TP_pixellib)/(TP_pixellib+FN_pixellib)
    precision_pixellib = (TP_pixellib)/(TP_pixellib+FP_pixellib)
    f1_pixellib = (2*precision_pixellib*recall_pixellib)/(precision_pixellib+recall_pixellib)

    print('-----result-----')
    print('yolov5')
    print(f'acc: {acc_yolov5}')
    print(f'recall: {recall_yolov5}')
    print(f'precision: {precision_yolov5}')
    print(f'f1-score: {f1_yolov5}')

    print('\npixellib')
    print(f'acc: {acc_pixellib}')
    print(f'recall: {recall_pixellib}')
    print(f'precision: {precision_pixellib}')
    print(f'f1-score: {f1_pixellib}')

    # plt.plot(len(count_yolov5), acc, label="line 1")
    # plt.plot(len(count_yolov5), x, label="line 2")
    # plt.plot(x, np.sin(x), label="curve 1")
    # plt.plot(x, np.cos(x), label="curve 2")
    # plt.legend()
    # plt.show()