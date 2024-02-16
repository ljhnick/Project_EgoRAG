import json

video_uids_file = open('data/val_video_ids_all.txt', 'r')
video_uids = video_uids_file.read().splitlines()

nlq_anno_path = 'data/ego4d_data/v2/annotations/nlq_val.json'
narration_path = 'data/ego4d_data/v2/annotations/narration.json'

nlq_anno = json.load(open(nlq_anno_path, 'r'))
narration = json.load(open(narration_path, 'r'))

video_data = {}

for video_uid in video_uids:
    video_data[video_uid] = {}

# iterate through nlq_anno video_uids
# extract query and ground truth from nlq_anno    
for video in nlq_anno['videos']:
    uid = video['video_uid']
    if uid in video_data:
        video_data[uid]['annotations'] = []
        for clip in video['clips']:
            for anno in clip['annotations']:
                for query in anno['language_queries']:
                    q = query.copy()
                    q['clip_uid'] = clip['clip_uid']
                    q['clip_start_frame'] = clip['video_start_frame']
                    q['clip_end_frame'] = clip['video_end_frame']
                    video_data[uid]['annotations'].append(q)

# extract narration from narration
keys_to_pop = []
for key in video_data:
    if narration[key]['status'] == 'redacted':
        keys_to_pop.append(key)
        continue
    nar = narration[key]['narration_pass_2']['narrations']
    video_data[key]['narration'] = nar

video_data = {k: video_data[k] for k in video_data if k not in keys_to_pop}

# save as json file
with open('data/data_process_val_all.json', 'w') as f:
    json.dump(video_data, f)
    

print('done')