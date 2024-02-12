import argparse
import json
import pandas as pd
from EgoRAG.egoRAG import EgoRAG
from tools.eval_utils import check_if_overlap, check_iou


parser = argparse.ArgumentParser(description='Run the pipeline')
args = parser.add_argument('--video_uids', type=str, help='The file containing the video uids')
args = parser.add_argument('--video_folder_path', type=str, default='/Users/jiahaoli/Library/CloudStorage/Dropbox/02_Career/Projects/Project_EgoRAG/data/ego4d_data/v2/full_scale')
args = parser.add_argument('--annotation_path', type=str, default='/Users/jiahaoli/Library/CloudStorage/Dropbox/02_Career/Projects/Project_EgoRAG/data/data_process_val_all.json')
args = parser.add_argument('--result_path', type=str, default='/Users/jiahaoli/Library/CloudStorage/Dropbox/02_Career/Projects/Project_EgoRAG/data/processed/processed_results_multimodal_20_frames.csv')
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
    video_uids_path = '/Users/jiahaoli/Library/CloudStorage/Dropbox/02_Career/Projects/Project_EgoRAG/data/video_uids_top10_spatial_queries.txt'
    # video_uids = ['18a3840b-7463-43c4-9aa9-b1d8e486fa84']
    video_uids = open(video_uids_path, 'r').read().split('\n')

    result_df = None
    egoRAG = EgoRAG(video_folder_path=args.video_folder_path, annotation_path=args.annotation_path, processed_saving_path=args.result_path)
    for video_uid in video_uids:
        egoRAG.load_video(video_uid)
        processed_result = egoRAG.run()
        if result_df is None:
            result_df = processed_result
        else:
            result_df = pd.concat([result_df, processed_result], ignore_index=True)

    result_df.to_csv(args.result_path, index=False)
    
    eval_result(result_df)

def test_results():
    result_path = '/Users/jiahaoli/Library/CloudStorage/Dropbox/02_Career/Projects/Project_EgoRAG/data/processed/processed_results_multimodal_top_10.csv'
    result_df = pd.read_csv(result_path)

    grouped = result_df.groupby('video_uid')

    for video_uid, group in grouped:
        overlap, iou_03 = eval_result(group)
        print(f'Video {video_uid}: Overlap: {overlap}, IoU 0.3: {iou_03}')
    overlap, iou_03 = eval_result(result_df)
    print(f'Overall: Overlap: {overlap}, IoU 0.3: {iou_03}')

if __name__ == '__main__':
    # main()
    test_results()