import torch
import torchvision
from torchvision import transforms as T
import cv2
import cvzone
import os
import sys

YOLOV5 = 0


if YOLOV5:
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
else:
    model = torchvision.models.detection.ssd300_vgg16(pretrained = True)
    # model = torchvision.models.detection.retinanet_resnet50_fpn(pretrained = True)
    model.eval()

classes_file = os.path.join(sys.path[0], "classes.txt")
classnames = []
with open(classes_file,'r') as f:
    classnames = f.read().splitlines()

# print(classnames[0])

image_filename = os.path.join(sys.path[0], "pedestrians.jpg")

if YOLOV5:

    # Inference
    results = model(image_filename)

    # Results
    results.print()
    results.save()  # or .show()

    results.xyxy[0]  # img1 predictions (tensor)
    rs = results.pandas().xyxy[0]  # img1 predictions (pandas)
    print(rs)
else:
    image = cv2.imread(image_filename)
    img = image.copy()
    print(type(image))

    imgtransform = T.ToTensor()
    image = imgtransform(image)
    print(type(image))

    with torch.no_grad():
        ypred = model([image])
        print(ypred[0].keys())

        bbox,scores,labels = ypred[0]['boxes'],ypred[0]['scores'],ypred[0]['labels']
        nums = torch.argwhere(scores > 0.30).shape[0]
        for i in range(nums):
            x,y,w,h = bbox[i].numpy().astype('int')
            cv2.rectangle(img,(x,y),(w,h),(0,0,255),3)
            classname = labels[i].numpy().astype('int')
            classdetected = classnames[classname-1]
            text = classdetected + " " + str(scores[i].numpy())
            cvzone.putTextRect(img,text,[x,y+100],scale=1,border=1,thickness=2)

    cv2.imshow('frame',img)
    cv2.waitKey(0)
