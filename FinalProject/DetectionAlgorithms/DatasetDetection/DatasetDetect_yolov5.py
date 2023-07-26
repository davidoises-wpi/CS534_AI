import os
import sys
from pathlib import Path
import torch
import cv2

FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]  # YOLOv5 root directory
YOLO_ROOT = ROOT / 'yolov5'
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if str(YOLO_ROOT) not in sys.path:
    sys.path.append(str(YOLO_ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.general import check_img_size
from utils.dataloaders import LoadImages
from utils.general import Profile
from utils.general import non_max_suppression
from utils.general import scale_boxes
from utils.plots import Annotator, colors

source = str(ROOT / 'images/pedestrian-walk-car-waiting.jpg')

save_dir = Path(ROOT / 'runs')
save_dir.mkdir(parents=True, exist_ok=True)

device = torch.device('cpu')

weights = ROOT / 'yolov5/yolov5s.pt'
dnn = False
data = ROOT / 'data/coco128.yaml'
half=False
imgsz=(640, 640)

model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
stride, names, pt = model.stride, model.names, model.pt
imgsz = check_img_size(imgsz, s=stride)  # check image size

bs = 1  # batch_size

model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup

def detect(image_filename):

    dataset = LoadImages(image_filename, img_size=imgsz, stride=stride, auto=pt, vid_stride=1)

    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())

    detections = []

    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.float()
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            pred = model(im, augment=False, visualize=False)

        # NMS
        with dt[2]:
            conf_thres=0.2 # confidence threshold
            iou_thres=0.45  # NMS IOU threshold
            classes=None # filter by class: --class 0, or --class 0 2 3
            agnostic_nms=False  # class-agnostic NMS
            max_det=1000  # maximum detections per image
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            im0 = im0s.copy()

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)  # integer class
                    label = names[c]
                    if label == "person":
                        label = "pedestrian"

                        det = {'x0': float(xyxy[0]),
                            'x1': float(xyxy[2]),
                            'y0': float(xyxy[1]),
                            'y1': float(xyxy[3]),
                            'score': float(conf),
                            'identity': label,
                            'orient': 0.0}

                        detections.append(det)

    return detections

def main():

    image_filename = os.path.join(Path(sys.path[0]).parents[1], "images/amsterdam_01078.png")
    detect(image_filename)

if __name__ == "__main__":
    main()
