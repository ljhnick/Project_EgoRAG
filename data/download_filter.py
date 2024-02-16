from pathlib import Path
import json

root = Path(__file__).parent.parent

val_annotation_path = root / 'data/ego4d_data/v2/annotations/nlq_val.json'

val_data = json.load(val_annotation_path.open('r'))

video_ids = []

for video in val_data['videos']:
    video_ids.append(video['video_uid'])

# save first 50 elements in video_ids into a text file
with open('data/val_video_ids_all.txt', 'w') as f:
    for item in video_ids:
        f.write("%s\n" % item)

