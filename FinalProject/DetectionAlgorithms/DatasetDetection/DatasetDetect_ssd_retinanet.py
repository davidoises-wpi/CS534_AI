import torch
import torchvision
from torchvision import transforms as T
from torchvision.models.detection.ssd import SSD300_VGG16_Weights
import cv2
import cvzone
import os
import sys
from pathlib import Path

# project_root = Path(sys.path[0]).parents[1]

classes_file = os.path.join(Path(sys.path[0]).parents[1], "classes.txt")
classnames = []
with open(classes_file,'r') as f:
    classnames = f.read().splitlines()

def detect(image_filename, algorithm='ssd'):

    if algorithm == 'ssd':
        # model = torchvision.models.detection.ssd300_vgg16(pretrained = True)
        model = torchvision.models.detection.ssd300_vgg16(weights=SSD300_VGG16_Weights.DEFAULT)
        # model = torchvision.models.detection.ssd300_vgg16(weights=SSD300_VGG16_Weights.COCO_V1)
    else:
        model = torchvision.models.detection.retinanet_resnet50_fpn(pretrained = True)
    model.eval()

    image = cv2.imread(image_filename)
    imgtransform = T.ToTensor()
    image = imgtransform(image)

    with torch.no_grad():
        ypred = model([image])

        bbox,scores,labels = ypred[0]['boxes'],ypred[0]['scores'],ypred[0]['labels']
        nums = torch.argwhere(scores > 0.30).shape[0]

        detections = []

        for i in range(nums):
            x0,y0,x1,y1 = bbox[i].numpy().astype('int')
            classname = labels[i].numpy().astype('int')
            classdetected = classnames[classname-1]
            score = str(scores[i].numpy())

            if classdetected == "person":

                classdetected = "pedestrian"
                det = {'x0': float(x0),
                       'x1': float(x1),
                       'y0': float(y0),
                       'y1': float(y1),
                       'score': score,
                       'identity': classdetected,
                       'orient': 0.0}

                detections.append(det)

    return detections

def main():

    # image_filename = os.path.join(Path(sys.path[0]).parents[1], "images/pedestrians.jpg")
    image_filename = os.path.join(Path(sys.path[0]).parents[1], "images/amsterdam_01078.png")
    detect(image_filename, algorithm='ssd')

if __name__ == "__main__":
    main()
