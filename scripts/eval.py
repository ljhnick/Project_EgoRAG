import pandas as pd 

from tools.eval_utils import check_if_overlap, check_iou


result_1 = pd.read_csv('/Users/jiahaoli/Library/CloudStorage/Dropbox/02_Career/Projects/Project_EgoRAG/data/processed/processed_results_multimodal_gt_text_only_20240214-224232.csv')
result_2 = pd.read_csv('/Users/jiahaoli/Library/CloudStorage/Dropbox/02_Career/Projects/Project_EgoRAG/data/processed/multimodal_gt_text_only_all_videos_20240215-193133.csv')

result = pd.concat([result_1, result_2])

print(f'Lenght of result: {len(result)}')

metric_overlap_count = 0
metric_iou_03_count = 0
total_count = 0

for index, row in result.iterrows():
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

print(f'Overlap: {overlap}, IoU 0.3: {iou_03}')