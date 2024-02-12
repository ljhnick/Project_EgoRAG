from EgoRAG.datasets.video_loader import Ego4DDataLoader
from EgoRAG.llm.wrapper import GPTWrapper

import matplotlib.pyplot as plt
import json
from tqdm import tqdm
import pandas as pd

def parse_result(result):
    # extract two numbers from the string
    pred = result['predictions']
    try:
        pred = eval(pred)
    except:
        pred_start = 0
        pred_end = 0
        
    if isinstance(pred, list) and len(pred) > 0:
        pred_start = pred[0]
        pred_end = pred[1]
    else:
        pred_start = 0
        pred_end = 0

    return pred_start, pred_end


class EgoRAG():
    def __init__(self,
                 video_folder_path,
                 annotation_path,
                 processed_saving_path):
        self.video_folder_path = video_folder_path
        self.processed_saving_path = processed_saving_path
        self.video_loader = Ego4DDataLoader(video_folder_path=video_folder_path, annotation_path=annotation_path)
        self.llm = GPTWrapper()

    def __len__(self):
        return len(self.video_paths)

    def load_video(self, video_uid):
        self.video_loader.load_video(video_uid=video_uid)
        self.video_data = self.video_loader.data_dict # {'video_uid': video_uid, 'gt_caption': gt_captions, 'anno': anno}

    def check_captions(self):
        return True
    
    def generate_captions(self):
        return True

    def generate_keyframes_captions_pairs(self, captions, gt_start_frame, gt_end_frame, clip_start, clip_end, total_keyframes=10):
        gt_keyframe = int((gt_start_frame + gt_end_frame) / 2)
        interval = (clip_end - clip_start) // total_keyframes
        keyframes = []

        current_frame = gt_keyframe
        while current_frame >= clip_start:
            keyframes.append(current_frame)
            current_frame -= interval
        current_frame = gt_keyframe + interval
        while current_frame <= clip_end:
            keyframes.append(current_frame)
            current_frame += interval

        captions_multimodal = captions.copy()
        for cap in captions_multimodal:
            cap['keyframe'] = None
        
        for keyframe in keyframes:
            cap_to_add = min(captions_multimodal, key=lambda x: abs(x['timestamp_frame'] - keyframe))
            keyframe_image = self.video_loader._get_video_frame_at_timestamp(keyframe)
            cap_to_add['keyframe'] = keyframe_image
        
        return captions_multimodal


    def run(self):
        anno = self.video_data['anno']
        captions = self.video_data['gt_caption'] # list of strings

        processed_result = []

        for index, row in tqdm(anno.iterrows(), total=anno.shape[0]):
            # print(index)
            # if index > 5:
            #     break
            query = row['query']
            clip_start_frame = row['clip_start_frame']
            clip_end_frame = row['clip_end_frame']
            video_start_frame = row['video_start_frame']
            video_end_frame = row['video_end_frame']

            multimodal_input = self.generate_keyframes_captions_pairs(captions, video_start_frame, video_end_frame, clip_start_frame, clip_end_frame, total_keyframes=20)

            prompt = self.llm.generate_prompt(query, multimodal_input, clip_start_frame, clip_end_frame)
            is_sucess, result = self.llm.call_gpt_api_text_and_image(prompt)
            if not is_sucess:
                continue
            result = result.choices[0].message.content
            result_json = result.split('{')[1].split('}')[0]
            result_json = json.loads('{' + result_json + '}')

            result_parsed = parse_result(result_json)

            processed_entry = {}
            processed_entry['video_uid'] = self.video_data['video_uid']
            processed_entry['query'] = query
            processed_entry['clip_start_frame'] = clip_start_frame
            processed_entry['clip_end_frame'] = clip_end_frame
            processed_entry['gt_start_frame'] = video_start_frame
            processed_entry['gt_end_frame'] = video_end_frame
            processed_entry['pred_start_frame'] = result_parsed[0]
            processed_entry['pred_end_frame'] = result_parsed[1]

            processed_result.append(processed_entry)
        
        processed_res = pd.DataFrame(processed_result)
        # processed_res.to_csv(self.processed_saving_path, index=False)

        return processed_res