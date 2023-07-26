import json
import os
import glob
import DatasetDetect_ssd_retinanet
from pathlib import Path
import sys

FILE = Path(__file__).resolve()
ECPB_ROOT = FILE.parents[2] / 'ECPB'
if str(ECPB_ROOT) not in sys.path:
    sys.path.append(str(ECPB_ROOT))

import dataconverter

def detector(image):
    detections = DatasetDetect_ssd_retinanet.detect(image, algorithm='ssd')
    return detections


def run_detector_on_dataset(root_folder, time='day', mode='val'):
    assert mode in ['val', 'test']
    assert time in ['day', 'night']

    # eval_imgs = glob.glob('./data/{}/img/{}/*/*'.format(time, mode))
    # eval_imgs = glob.glob(os.path.join(root_folder, 'ECPB/data/{}/img/{}/amsterdam/amsterdam_01078.png'.format(time, mode)))
    eval_imgs = glob.glob(os.path.join(root_folder, 'ECPB/data/{}/img/{}/*/*'.format(time, mode)))
    destdir = os.path.join(root_folder, 'results/{}/{}/'.format(time, mode))
    dataconverter.create_base_dir(destdir)

    for im in eval_imgs:
        detections = detector(im)
        destfile = os.path.join(destdir, os.path.basename(im).replace('.png', '.json'))
        frame = {'identity': 'frame'}
        frame['children'] = detections
        json.dump(frame, open(destfile, 'w'), indent=1)


if __name__ == "__main__":

    project_root = Path(sys.path[0]).parents[1]
    run_detector_on_dataset(project_root, time='day', mode='val')
