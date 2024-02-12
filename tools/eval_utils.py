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