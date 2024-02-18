from EgoRAG.datasets.video_loader import Ego4DDataLoader
from EgoRAG.llm.wrapper import GPTWrapper

import matplotlib.pyplot as plt
import json
from tqdm import tqdm
import pandas as pd
import random

def parse_result(result):
    # extract two numbers from the string
    pred = result['predictions']
    pred_result = []
    is_init = False
    number = ''
    for char in pred:
        if char in '0123456789':
            number += char
            is_init = True
        elif is_init:
            pred_result.append(int(number))
            number = ''
            is_init = False
    try:
        pred_start = pred_result[0]
        pred_end = pred_result[1]
    except:
        pred_start = 0
        pred_end = 0

    # try:
    #     pred = eval(pred)
    # except:
    #     pred_start = 0
    #     pred_end = 0
        
    # if isinstance(pred, list) and len(pred) > 0:
    #     pred_start = pred[0]
    #     pred_end = pred[1]
    # else:
    #     pred_start = 0
    #     pred_end = 0
            
    

    return pred_start, pred_end


class EgoRAG():
    def __init__(self,
                 video_folder_path,
                 annotation_path,
                 processed_saving_path,
                 load_result=False,
                 text_only=False,
                 spatial_text=False):
        self.video_folder_path = video_folder_path
        self.processed_saving_path = processed_saving_path
        self.is_load_result = load_result
        self.video_loader = Ego4DDataLoader(video_folder_path=video_folder_path, annotation_path=annotation_path)
        self.llm = GPTWrapper()
        self.text_only = text_only
        self.spatial_text = spatial_text

    def __len__(self):
        return len(self.video_paths)

    def load_video(self, video_uid):
        self.video_loader.load_video(video_uid=video_uid)
        self.video_data = self.video_loader.data_dict # {'video_uid': video_uid, 'gt_caption': gt_captions, 'anno': anno}

    def check_captions(self):
        return True
    
    def generate_captions(self, clip_length_sec=2):
        raw_captions = self.video_data['gt_caption']

        interval = int(clip_length_sec * self.video_loader.fps / 2)
        captions_processed = []
        for index, caption in enumerate(raw_captions):
            caption_processed = {}

            text = caption['narration_text']
            timestamp = caption['timestamp_frame']
            pre_timestamp = raw_captions[index - 1]['timestamp_frame'] if index > 0 else 0
            post_timestamp = raw_captions[index + 1]['timestamp_frame'] if index < len(raw_captions) - 1 else raw_captions[-1]['timestamp_frame']

            # start_frame = timestamp - interval if timestamp - interval > int((timestamp + pre_timestamp)/2) else int((timestamp + pre_timestamp)/2)
            start_frame = int((timestamp + pre_timestamp)/2)
            start_frame = max(start_frame, 0)

            # end_frame = timestamp + interval if timestamp + interval < int((timestamp + post_timestamp)/2) else int((timestamp + post_timestamp)/2)
            end_frame = int((timestamp + post_timestamp)/2)
            end_frame = min(end_frame, raw_captions[-1]['timestamp_frame'])

            caption_processed['caption'] = text
            caption_processed['timestamp_frame'] = timestamp
            caption_processed['start_frame'] = start_frame
            caption_processed['end_frame'] = end_frame

            captions_processed.append(caption_processed)
            
            self.captions_processed = captions_processed

        return captions_processed


        # target output format {start_frame, end_frame, caption}
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
            if self.spatial_text:
                cap_to_add['keyframe'] = self.llm.generate_spatial_description(keyframe_image)
            else:
                cap_to_add['keyframe'] = keyframe_image
        
        return captions_multimodal
    

    def generate_multimodal_input(self, captions, gt_start_frame, gt_end_frame, total_pairs=15):
        gt_keyframe = int((gt_start_frame + gt_end_frame) / 2)

        index_gt = min(range(len(captions)), key=lambda i: abs(captions[i]['timestamp_frame'] - gt_keyframe))
        start_index = max(0, index_gt - total_pairs//2)
        end_index = min(len(captions), start_index + total_pairs)

        captions_multimodal = captions[start_index:end_index]
        for index, cap in enumerate(captions_multimodal):
            caption_keyframe = self.video_loader._get_video_frame_at_timestamp(cap['timestamp_frame'])
            if self.spatial_text:
                is_success, result = self.llm.generate_spatial_description(caption_keyframe)
                if is_success:
                    result = result.choices[0].message.content
                    captions_multimodal[index]['spatial_description'] = result
                    # print(result)
            else:
                captions_multimodal[index]['spatial_description'] = ''
            captions_multimodal[index]['keyframe'] = caption_keyframe



        return captions_multimodal
        

    def run(self):
        anno = self.video_data['anno']
        # captions = self.video_data['gt_caption'] # list of strings
        captions = self.generate_captions()

        processed_result = []

        for index, row in tqdm(anno.iterrows(), total=len(anno)):
            # print(index)
            # if index > 5:
            #     break
            query = row['query']
            clip_start_frame = row['clip_start_frame']
            clip_end_frame = row['clip_end_frame']
            video_start_frame = row['video_start_frame']
            video_end_frame = row['video_end_frame']
            template = row['template']

            if not isinstance(query, str):
                continue

            # wrong
            # multimodal_input = self.generate_keyframes_captions_pairs(captions, video_start_frame, video_end_frame, clip_start_frame, clip_end_frame, total_keyframes=10)

            multimodal_input = self.generate_multimodal_input(captions, video_start_frame, video_end_frame, total_pairs=15)

            prompt = self.llm.generate_prompt(query, multimodal_input, clip_start_frame, clip_end_frame, text_only=self.text_only, spatial_text=self.spatial_text)

            if self.spatial_text or self.text_only:
                is_sucess, result = self.llm.call_gpt_api_text_only(prompt)
            else:
                is_sucess, result = self.llm.call_gpt_api_text_and_image(prompt)
            if not is_sucess:
                continue

            try:
                result = result.choices[0].message.content
                result_json = result.split('{')[1].split('}')[0]
                result_json = json.loads('{' + result_json + '}')

                result_parsed = parse_result(result_json)

            except Exception as e:
                print(e)
                continue

            processed_entry = {}
            processed_entry['video_uid'] = self.video_data['video_uid']
            processed_entry['query'] = query
            processed_entry['template'] = template
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
    
    def update_result(self):
        anno = self.video_data['anno']
        uid = self.video_data['video_uid']
        result = pd.read_csv(self.processed_saving_path)
        result = result[result['video_uid'] == uid]
    

        for index, row in result.iterrows():
            query = row['query']

            template = anno[anno['query'] == query]['template'].values[0]
            result.at[index, 'template'] = template


        return result