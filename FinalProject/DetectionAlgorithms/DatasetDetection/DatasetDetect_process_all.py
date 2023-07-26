import json
import os
import glob
import DatasetDetect_ssd
import DatasetDetect_retinanet
import DatasetDetect_yolov5
from pathlib import Path
import sys
import time

FILE = Path(__file__).resolve()
ECPB_ROOT = FILE.parents[2] / 'ECPB'
if str(ECPB_ROOT) not in sys.path:
    sys.path.append(str(ECPB_ROOT))

import dataconverter

image_count_limit = 1000

def run_detector_on_dataset(root_folder, time_of_day='day', mode='val', algorithm='ssd'):
    assert mode in ['val', 'test']
    assert time_of_day in ['day', 'night']

    # eval_imgs = glob.glob(os.path.join(root_folder, 'ECPB/data/{}/img/{}/amsterdam/amsterdam_01078.png'.format(time, mode)))
    eval_imgs = glob.glob(os.path.join(root_folder, 'ECPB/data/{}/img/{}/*/*'.format(time_of_day, mode)))
    destdir = os.path.join(root_folder, 'results/{}/'.format(algorithm))
    dataconverter.create_base_dir(destdir)

    execution_times = []
    image_counter = 0
    for im in eval_imgs:

        st = time.time()
        if algorithm == 'yolov5':
            detections = DatasetDetect_yolov5.detect(im)
        elif algorithm == 'retinanet':
            detections = DatasetDetect_retinanet.detect(im)
        else:
            detections = DatasetDetect_ssd.detect(im)
        et = time.time()

        execution_times.append((et-st)*1000.0)

        destfile = os.path.join(destdir, os.path.basename(im).replace('.png', '.json'))
        frame = {'identity': 'frame'}
        frame['children'] = detections
        json.dump(frame, open(destfile, 'w'), indent=1)

        image_counter = image_counter + 1

        if image_counter == image_count_limit:
            break

    sum_of_times = 0
    for i in range(len(execution_times)):
        sum_of_times = sum_of_times + execution_times[i]
    average_execution_time = sum_of_times/len(execution_times)

    execution_time_log = os.path.join(destdir, 'execution_time.txt')
    f= open(execution_time_log, "w")
    f.write("Average execution time %f\r\n" % average_execution_time)

if __name__ == "__main__":

    project_root = Path(sys.path[0]).parents[1]

    run_detector_on_dataset(project_root, time_of_day='day', mode='val', algorithm='ssd')

    run_detector_on_dataset(project_root, time_of_day='day', mode='val', algorithm='retinanet')

    run_detector_on_dataset(project_root, time_of_day='day', mode='val', algorithm='yolov5')
