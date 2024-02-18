import argparse
import json
import os
import pandas as pd
from EgoRAG.egoRAG import EgoRAG
from tools.eval_utils import check_if_overlap, check_iou
from datetime import datetime
from tqdm import tqdm

current_time = datetime.now().strftime("%Y%m%d-%H%M%S")

parser = argparse.ArgumentParser(description='Run the pipeline')
args = parser.add_argument('--video_uids', type=str, help='The file containing the video uids')
args = parser.add_argument('--video_folder_path', type=str, default='/home/nick/Research/Project_EgoRAG/data/ego4d_data/v2/full_scale')
args = parser.add_argument('--annotation_path', type=str, default='/home/nick/Research/Project_EgoRAG/data/data_process_val_all.json')
args = parser.add_argument('--result_path', type=str, default=f'/home/nick/Research/Project_EgoRAG/data/processed/multimodal_gt_text_only_all_videos_{current_time}.csv')
args = parser.add_argument('--text_only', type=bool, default=True)
args = parser.parse_args()

def eval_result(result_pd):

    metric_overlap_count = 0
    metric_iou_03_count = 0
    total_count = 0

    for index, row in result_pd.iterrows():
        total_count += 1

        gt_start = row['gt_start_frame']
        gt_end = row['gt_end_frame']
        pred_start = row['pred_start_frame']
        pred_end = row['pred_end_frame']

        try:
            gt_start = int(gt_start)
            gt_end = int(gt_end)
            pred_start = int(pred_start)
            pred_end = int(pred_end)
        except:
            continue

        if check_if_overlap(gt_start, gt_end, pred_start, pred_end):
            metric_overlap_count += 1
        if check_iou(gt_start, gt_end, pred_start, pred_end, 0.3):
            metric_iou_03_count += 1
    
    overlap = metric_overlap_count / total_count
    iou_03 = metric_iou_03_count / total_count
    
    return overlap, iou_03

def main():
    # 1. load video using uids; 2. load captions (either from ground truth or running caption model or from existed results)
    video_uids_path = '/home/nick/Research/Project_EgoRAG/data/val_video_ids_all.txt'
    video_uids = open(video_uids_path, 'r').read().split('\n')

    temp_path = '/home/nick/Research/Project_EgoRAG/data/processed/temp'
    processed_videos = os.listdir(temp_path)
    processed_video_uids = []
    for video in processed_videos:
        processed_video_uids.append(video.split('.')[0])
    # existed_result = '/Users/jiahaoli/Library/CloudStorage/Dropbox/02_Career/Projects/Project_EgoRAG/data/processed/processed_results_multimodal_gt_keyframe_20240214-205439.csv'
    # existed_result = pd.read_csv(existed_result)
    # existed_uids = existed_result['video_uid'].unique()

    result_df = None
    egoRAG = EgoRAG(video_folder_path=args.video_folder_path, annotation_path=args.annotation_path, processed_saving_path=args.result_path, text_only=args.text_only, spatial_text=False)

    error_uids = []
    for video_uid in tqdm(video_uids):
        if video_uid in processed_video_uids:
            continue
        try:
            egoRAG.load_video(video_uid)
        except Exception as e:
            error_uids.append(video_uid)
            print(f'Error loading video {video_uid}')
            print(e)
            continue
        # try:
        processed_result = egoRAG.run()
        # except Exception as e:
        #     error_uids.append(video_uid)
        #     print(f'Error processing video {video_uid}')
        #     print(e)
        #     continue
        # processed_result = egoRAG.update_result()

        temp_save_path = '/home/nick/Research/Project_EgoRAG/data/processed/temp/'
        temp_save_path = temp_save_path + video_uid + '.csv'
        processed_result.to_csv(temp_save_path, index=False)
        if result_df is None:
            result_df = processed_result
        else:
            result_df = pd.concat([result_df, processed_result], ignore_index=True)

    result_df.to_csv(args.result_path, index=False)

    overlap, iou_03 = eval_result(result_df)
    print(f'Overall: Overlap: {overlap}, IoU 0.3: {iou_03}')


def test_results():
    result_path = '/Users/jiahaoli/Library/CloudStorage/Dropbox/02_Career/Projects/Project_EgoRAG/data/processed/processed_results_multimodal_gt_keyframe_20240214-205439.csv'
    result_df = pd.read_csv(result_path)

    result_text_only_path = '/Users/jiahaoli/Library/CloudStorage/Dropbox/02_Career/Projects/Project_EgoRAG/data/ego4d_narration_results.csv'
    result_text_df = pd.read_csv(result_text_only_path)

    template_spatial = [
        # 'Objects: Where is object X before / after event Y?',
        # 'Place: Where did I put X?',
        'Objects: Where is object X?',
        'Objects: In what location did I see object X ?',
        # 'Objects: Where is my object X?'
    ]
    result_df = result_df[result_df['template'].isin(template_spatial)]

    print(len(result_df))

    grouped = result_df.groupby('video_uid')

    for video_uid, group in grouped:
        overlap, iou_03 = eval_result(group)
        print(f'Video {video_uid}: Overlap: {overlap}, IoU 0.3: {iou_03}')
    overlap, iou_03 = eval_result(result_df)
    print(f'Overall: Overlap: {overlap}, IoU 0.3: {iou_03}')

if __name__ == '__main__':
    main()
    # test_results()