import json
import argparse
import pandas as pd

def check_if_overlap(gt_start, gt_end, predicted_start, predcited_end):
    if gt_start > predcited_end or gt_end < predicted_start:
        return False
    return True   

def check_if_correct(gt_start, gt_end, predicted_start, predcited_end):
    if gt_start >= predicted_start and gt_end <= predcited_end:
        return True
    return False


def check_iou(gt_start, gt_end, predicted_start, predcited_end, threshold=0.3):
    intersection = max(0, min(gt_end, predcited_end) - max(gt_start, predicted_start))
    union = min(max(gt_end, predcited_end) - min(gt_start, predicted_start), gt_end - gt_start + predcited_end - predicted_start)
    if union == 0:
        return False
    iou = intersection / union
    if iou >= threshold:
        return True
    return False

def compute_metrics(data_eval, video_uid):
    metric_overlap_count = 0
    metric_iou_03_count = 0
    na_count = 0
    total_count = 0
    total_count_no_na = 0
    result = data_eval[video_uid]['result']

    if len(result) < 10:
        return video_uid, 0, False
    
    for entry in result:
        gt_start = entry['gt_start']
        gt_end = entry['gt_end']

        predicted_start = entry['predicted_start']
        if not isinstance(predicted_start, int):
            print('predicted_start is not int')
            continue

        predicted_start = entry['predicted_start']
        predicted_end = entry['predicted_end']

        isna = entry['NA']

        if isna:
            na_count += 1
        else:
            total_count_no_na += 1
            if check_if_overlap(gt_start, gt_end, predicted_start, predicted_end):
                metric_overlap_count += 1
            if check_iou(gt_start, gt_end, predicted_start, predicted_end) > 0.3:
                metric_iou_03_count += 1

        total_count += 1

    if total_count_no_na == 0:
        return video_uid, 0, False
    
    iou_03 = metric_iou_03_count / total_count_no_na

    return video_uid, iou_03, True


def pick_best_10():
    data_1 = json.load(open('data/predicted_results_207.json', 'r'))
    data_2 = json.load(open('data/predicted_results_207_after.json', 'r'))

    data = []
    for video_uid in data_1:
        vid, iou_03, is_threshold = compute_metrics(data_1, video_uid)
        data_entry = {}
        data_entry['vid'] = vid
        data_entry['iou_03'] = iou_03
        data_entry['is_threshold'] = is_threshold
        data.append(data_entry)

    for video_uid in data_2:
        vid, iou_03, is_threshold = compute_metrics(data_2, video_uid)
        data_entry = {}
        data_entry['vid'] = vid
        data_entry['iou_03'] = iou_03
        data_entry['is_threshold'] = is_threshold
        data.append(data_entry)

    filtered_data = [d for d in data if d['is_threshold']]
    sorted_data = sorted(filtered_data, key=lambda x: x['iou_03'])

    top_5 = sorted_data[:5]
    lowest_5 = sorted_data[-5:]

    # save the vid for top 5 and lowest 5
    top_5_vid = [d['vid'] for d in top_5]
    lowest_5_vid = [d['vid'] for d in lowest_5]

    with open('data/top_5_vid.json', 'w') as f:
        json.dump(top_5_vid, f)
    with open('data/lowest_5_vid.json', 'w') as f:
        json.dump(lowest_5_vid, f)


def eval_spatial():
    data_eval = pd.read_csv(args.data)
    template_spatial = [
        'Objects: Where is object X before / after event Y?',
        'Place: Where did I put X?',
        'Objects: Where is object X?',
        'Objects: In what location did I see object X ?',
        'Objects: Where is my object X?'
    ]
    filtered_data = data_eval[data_eval['template'].isin(template_spatial)]
    # filtered_data = data_eval[~data_eval['template'].isin(template_spatial)]
    # filtered_data = data_eval

    target_vid_data = filtered_data[data_eval['video_uid'] == '18e84829-901a-414d-8a2b-d1d2b3244db7']
    print(len(target_vid_data))
    
    

    vid_top10 = filtered_data['video_uid'].value_counts().head(10).index.tolist()
    data_top10_spatial = filtered_data[filtered_data['video_uid'].isin(vid_top10)]

    print(len(data_top10_spatial))


    metric_overlap_count = 0
    metric_iou_03_count = 0
    metric_mean_r1 = 0
    na_count = 0
    total_count = 0
    total_count_no_na = 0

    for index, row in data_top10_spatial.iterrows():
        try:
            predict_start = int(row['predicted_start'])
            predict_end = int(row['predicted_end'])
            gt_start = int(row['gt_start'])
            gt_end = int(row['gt_end'])
            isna = row['is_NA']
        except Exception as e:
            print(e)
            continue

        total_count += 1

        if isna:
            na_count += 1
            continue
        total_count_no_na += 1

        if check_if_overlap(gt_start, gt_end, predict_start, predict_end):
            metric_overlap_count += 1
        if check_iou(gt_start, gt_end, predict_start, predict_end):
            metric_iou_03_count += 1
        if check_if_correct(gt_start, gt_end, predict_start, predict_end):
            metric_mean_r1 += 1
        
    na_rate = na_count / total_count
    overlap = metric_overlap_count / total_count
    iou_03 = metric_iou_03_count / total_count
    mean_r1 = metric_mean_r1 / total_count
    print(f'NA rate: {na_rate}')
    print(f'Overlap: {overlap}')
    print(f'IoU 0.3: {iou_03}')
    # print(f'Mean x@x: {mean_r1}')


parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='data/ego4d_narration_results.csv')
args = parser.parse_args()


def main():
    data_eval = json.load(open('data/predicted_results_207.json', 'r'))

    metric_overlap_count = 0
    metric_iou_03_count = 0
    na_count = 0
    total_count = 0
    total_count_no_na = 0

    for video_uid in data_eval:
        result = data_eval[video_uid]['result']

        for entry in result:
            gt_start = entry['gt_start']
            gt_end = entry['gt_end']

            predicted_start = entry['predicted_start']
            if not isinstance(predicted_start, int):
                print('predicted_start is not int')
                continue

            predicted_start = entry['predicted_start']
            predicted_end = entry['predicted_end']

            isna = entry['NA']

            if isna:
                na_count += 1
            else:
                total_count_no_na += 1
                if check_if_overlap(gt_start, gt_end, predicted_start, predicted_end):
                    metric_overlap_count += 1
                if check_iou(gt_start, gt_end, predicted_start, predicted_end) > 0.3:
                    metric_iou_03_count += 1

            total_count += 1

    na_rate = na_count / total_count
    overlap = metric_overlap_count / total_count_no_na
    iou_03 = metric_iou_03_count / total_count_no_na

    print(f'NA rate: {na_rate}')
    print(f'Overlap: {overlap}')
    print(f'IoU 0.3: {iou_03}')


if __name__ == '__main__':
    # main()
    eval_spatial()