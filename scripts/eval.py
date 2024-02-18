import os
import pandas as pd 

from tools.eval_utils import check_if_overlap, check_iou


# result_1 = pd.read_csv('/Users/jiahaoli/Library/CloudStorage/Dropbox/02_Career/Projects/Project_EgoRAG/data/processed/processed_results_multimodal_gt_text_only_20240214-224232.csv')
# result_2 = pd.read_csv('/Users/jiahaoli/Library/CloudStorage/Dropbox/02_Career/Projects/Project_EgoRAG/data/processed/multimodal_gt_text_only_all_videos_20240215-193133.csv')

# result = pd.concat([result_1, result_2])

result_path = '/home/nick/Research/Project_EgoRAG/data/processed/text_only'
result_all = os.listdir(result_path)

result = pd.DataFrame()

for result_file in result_all:
    try:
        result_temp = pd.read_csv(os.path.join(result_path, result_file))
        result = pd.concat([result, result_temp], ignore_index=True)
    except:
        continue

print(1)

template_spatial = [
    'Objects: Where is object X before / after event Y?',
    # 'Place: Where did I put X?',
    # 'Objects: Where is object X?',
    # 'Objects: In what location did I see object X ?',
    # 'Objects: Where is my object X?'
]

# result = result[result['template'].isin(template_spatial)]

print(f'Lenght of result: {len(result)}')

metric_overlap_count = 0
metric_iou_03_count = 0
total_count = 0

na_count = 0

for index, row in result.iterrows():
    

    gt_start = row['gt_start_frame']
    gt_end = row['gt_end_frame']
    pred_start = row['pred_start_frame']
    pred_end = row['pred_end_frame']

    if int(pred_start) == 0 and int(pred_end) == 0:
        na_count += 1
        continue

    total_count += 1

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
print(f'NA count: {na_count}')