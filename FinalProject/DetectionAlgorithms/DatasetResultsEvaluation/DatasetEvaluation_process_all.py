
import os
import sys
from pathlib import Path
import numpy as np

FILE = Path(__file__).resolve()
ECPB_ROOT = FILE.parents[2] / 'ECPB'
if str(ECPB_ROOT) not in sys.path:
    sys.path.append(str(ECPB_ROOT))

from dataloader import load_data_ecp
from params import ParamsFactory
from match import Evaluator
from params import IoU
from match import compare_all
import ECPB_statistics as statistics

def create_evaluator(data, difficulty, ignore_other_vru, type='pedestrian'):

    params = ParamsFactory(difficulty=difficulty,
                            ignore_other_vru=ignore_other_vru,
                            tolerated_other_classes=['rider'],
                            dont_care_classes=['person-group-far-away'],
                            detections_type=['pedestrian'],
                            ignore_type_for_skipped_gts=1,
                            size_limits={'reasonable': 40, 'small': 30,
                                        'occluded': 40, 'all': 20},
                            occ_limits={'reasonable': 40, 'small': 40,
                                        'occluded': 80, 'all': 80},
                            size_upper_limits={'small': 60},
                            occ_lower_limits={'occluded': 40},
                            discard_depictions=True,
                            clipping_boxes=True,
                            transform_det_to_xy_coordinates=True
                            )

    return Evaluator(data,
                      metric=IoU,
                      comparable_identities=compare_all,
                      ignore_gt=params.ignore_gt,
                      skip_gt=params.skip_gt,
                      skip_det=params.skip_det,
                      preprocess_gt=params.preprocess_gt,
                      preprocess_det=params.preprocess_det,
                      allow_multiple_matches=False)

def evaluate(difficulty, ignore_other_vru, results_path, det_path, gt_path, det_method_name,
             use_cache, eval_type='pedestrian'):
    """The actual evaluation"""

    # path to save pickled results
    pkl_path = os.path.join(results_path,
                            'ignore={}_difficulty={}_evaltype={}.pkl'.format(ignore_other_vru,
                                                                             difficulty, eval_type))

    data = load_data_ecp(gt_path, det_path)
    evaluator = create_evaluator(data, difficulty, ignore_other_vru, eval_type)
    result = evaluator.result

    # print('TP: ', result.tp)
    # print('FP: ', result.fp)
    # print('FN: ', result.nof_gts - result.tp)

    # recall = np.true_divide(result.tp, result.nof_gts)
    # precision = np.true_divide(result.tp, result.tp + result.fp)
    # print('recall: ', recall)
    # print('precision: ', precision)

    # Miss Rate vs False Positive Per Image
    mr_fppi = statistics.MrFppi(result=result)
    title = 'difficulty={}, ignore_other_vru={}, evaltype={}'.format(difficulty, ignore_other_vru,
                                                                     eval_type)
    label = 'lamr: {}'.format(mr_fppi.log_avg_mr_reference_implementation())
    fig = mr_fppi.create_plot(title, label)
    filename = 'lamr_ignore={}_difficulty={}_evaltype={}'.format(ignore_other_vru, difficulty,
                                                                 eval_type)

    fig.savefig(os.path.join(results_path, '{}.pdf'.format(filename)))  # vector graphic
    fig.savefig(os.path.join(results_path, '{}.png'.format(filename)))  # png

    print('# ----------------------------------------------------------------- #')
    print('Finished evaluation of ' + det_method_name)
    print('difficulty={}, ignore_other_vru={}, evaltype={}'.format(difficulty, ignore_other_vru,
                                                                   eval_type))
    print('---')
    print('Log-Avg Miss Rate (caltech reference implementation): ', \
        mr_fppi.log_avg_mr_reference_implementation())


def evaluate_detection(results_path, det_path, gt_path, det_method_name, eval_type='pedestrian'):
    print('Start evaluation for {}'.format(det_method_name))

    # Use the most generic settings
    difficulty = 'all'
    ignore_other_vru = False

    evaluate(difficulty, ignore_other_vru, results_path, det_path, gt_path, det_method_name,
                use_cache=False, eval_type=eval_type)

def eval(root_folder, time='day', mode='val', eval_type='pedestrian', det_method_name='ssd'):
    assert time in ['day', 'night']
    assert mode in ['val', 'test', 'working_subset']

    gt_path = str(root_folder / 'ECPB/data/{}/labels/{}'.format(time, mode))
    det_path = str(root_folder / 'results/{}'.format(det_method_name))

    # folder where you find all the results (unless you change other paths...)
    results_path = str(root_folder / 'results/{}/eval_results'.format(det_method_name))
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    evaluate_detection(results_path, det_path, gt_path, det_method_name, eval_type)
    print('')
    print('# -----------------------------------------------------------------')
    print('Finished evaluation, results can be found here: {}'.format(results_path))
    print('# -----------------------------------------------------------------')

if __name__ == "__main__":

    project_root = Path(sys.path[0]).parents[1]

    eval(project_root, time='day', mode='working_subset', eval_type='pedestrian', det_method_name='ssd')

    eval(project_root, time='day', mode='working_subset', eval_type='pedestrian', det_method_name='retinanet')

    eval(project_root, time='day', mode='working_subset', eval_type='pedestrian', det_method_name='yolov5')
