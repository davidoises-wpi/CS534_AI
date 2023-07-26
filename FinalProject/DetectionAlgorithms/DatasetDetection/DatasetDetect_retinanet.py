import torch
import torchvision
from torchvision import transforms as T
from torchvision.models.detection.retinanet import RetinaNet_ResNet50_FPN_Weights
import cv2
import os
import sys
from pathlib import Path

classes_file = os.path.join(Path(sys.path[0]).parents[1], "classes.txt")
classnames = []
with open(classes_file,'r') as f:
    classnames = f.read().splitlines()

model = torchvision.models.detection.retinanet_resnet50_fpn(weights=RetinaNet_ResNet50_FPN_Weights.DEFAULT)
model.eval()

def detect(image_filename):

    image = cv2.imread(image_filename)
    imgtransform = T.ToTensor()
    image = imgtransform(image)

    with torch.no_grad():
        ypred = model([image])

        bbox,scores,labels = ypred[0]['boxes'],ypred[0]['scores'],ypred[0]['labels']
        nums = torch.argwhere(scores > 0.20).shape[0]

        detections = []

        for i in range(nums):
            classname = labels[i].numpy().astype('int')
            classdetected = classnames[classname-1]

            if classdetected == "person":
                x0,y0,x1,y1 = bbox[i].numpy().astype('int')
                score = str(scores[i].numpy())
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

    image_filename = os.path.join(Path(sys.path[0]).parents[1], "images/amsterdam_01078.png")
    res = detect(image_filename)
    print(res)

if __name__ == "__main__":
    main()
