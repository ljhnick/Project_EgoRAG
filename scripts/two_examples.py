from EgoRAG.egoRAG import EgoRAG
from tools.eval_utils import check_if_overlap, check_iou

import pandas as pd
import argparse

parser = argparse.ArgumentParser(description='Run the pipeline')
args = parser.add_argument('--video_uids', type=str, help='The file containing the video uids')
args = parser.add_argument('--video_folder_path', type=str, default='/Users/jiahaoli/Library/CloudStorage/Dropbox/02_Career/Projects/Project_EgoRAG/data/ego4d_data/v2/full_scale')
args = parser.add_argument('--annotation_path', type=str, default='/Users/jiahaoli/Library/CloudStorage/Dropbox/02_Career/Projects/Project_EgoRAG/data/data_process_val_all.json')
args = parser.add_argument('--result_path', type=str, default='/Users/jiahaoli/Library/CloudStorage/Dropbox/02_Career/Projects/Project_EgoRAG/data/processed/processed_results_multimodal_top_10.csv')
args = parser.parse_args()

result_egoRAG = '/Users/jiahaoli/Library/CloudStorage/Dropbox/02_Career/Projects/Project_EgoRAG/data/processed/processed_results_multimodal_top_10.csv'
result_baseline = '/Users/jiahaoli/Library/CloudStorage/Dropbox/02_Career/Projects/Project_EgoRAG/data/ego4d_narration_results.csv'

result_egoRAG = pd.read_csv(result_egoRAG)
result_baseline = pd.read_csv(result_baseline)

template_spatial = [
    # 'Objects: Where is object X before / after event Y?',
    # 'Place: Where did I put X?',
    'Objects: Where is object X?',
    # 'Objects: In what location did I see object X ?',
    # 'Objects: Where is my object X?'
]

result_egoRAG = result_egoRAG[result_egoRAG['template'].isin(template_spatial)]

egoRAG = EgoRAG(video_folder_path=args.video_folder_path, annotation_path=args.annotation_path, processed_saving_path=args.result_path)

for index, row in result_egoRAG.iterrows():
    pre_start = row['pred_start_frame']
    pre_end = row['pred_end_frame']
    gt_start = row['gt_start_frame']
    gt_end = row['gt_end_frame']

    
