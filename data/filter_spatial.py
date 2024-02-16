import json
import pandas as pd

data = json.load(open('data/predicted_results.json', 'r'))

key_list = ['video_uid', 'clip_frames', 'query', 'template', 'is_NA', 'predicted_start', 'predicted_end', 'gt_start', 'gt_end']

data_json = {}
for key in key_list:
    data_json[key] = []


for video_uid in data:
    data_entry = data[video_uid]['result']
    for entry in data_entry:
        data_json['video_uid'].append(video_uid)
        data_json['clip_frames'].append(entry['clip_frames'])
        data_json['query'].append(entry['query'])
        data_json['template'].append(entry['template'])
        data_json['is_NA'].append(entry['NA'])
        data_json['predicted_start'].append(entry['predicted_start'])
        data_json['predicted_end'].append(entry['predicted_end'])
        data_json['gt_start'].append(entry['gt_start'])
        data_json['gt_end'].append(entry['gt_end'])


data_df = pd.DataFrame(data_json)

data_df.to_csv('data/ego4d_narration_results.csv', index=False)

print('done')

