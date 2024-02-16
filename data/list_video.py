import json

data_json = json.load(open('data/data_process_val_all.json', 'r'))

video_ids = data_json.keys()

# write to a txt file
with open('data/val_video_ids_all.txt', 'w') as f:
    for item in video_ids:
        f.write("%s\n" % item)