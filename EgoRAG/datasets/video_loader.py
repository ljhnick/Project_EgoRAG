# from torch.utils.data import Dataset

# 1. load videos
# 2. load annotations including queries and captions
# 3. load existing data

import json
import os
import cv2
import pandas as pd

class Ego4DDataLoader():

    template_spatial = [
        'Objects: Where is object X before / after event Y?',
        'Place: Where did I put X?',
        'Objects: Where is object X?',
        'Objects: In what location did I see object X ?',
        'Objects: Where is my object X?'
    ]

    def __init__(self,
                 video_folder_path,
                 annotation_path,
                 is_spatial_only=False):
        self.video_folder_path = video_folder_path
        self.annotation_path = annotation_path

        self.annotations_all = json.load(open(annotation_path, 'r'))
        self.data_dict = {}

        self.is_spaital_only = is_spatial_only
    
    def load_video(self, video_uid):
        # load the video
        self.video_uid = video_uid
        video_path = os.path.join(self.video_folder_path, video_uid + '.mp4')
        
        self.video_cap = cv2.VideoCapture(video_path)
        self.fps = self.video_cap.get(cv2.CAP_PROP_FPS)
        self.is_video_loaded = True

        # load annotation of the video
        annotation = self.annotations_all[video_uid]
        self.gt_captions = annotation['narration']
        self.anno = pd.DataFrame(annotation['annotations'])

        if self.is_spaital_only:
            self.anno = self.anno[self.anno['template'].isin(self.template_spatial)]

        self.data_dict['video_uid'] = video_uid
        self.data_dict['gt_caption'] = self.gt_captions
        self.data_dict['anno'] = self.anno


    def _get_video_frame_at_timestamp(self, timestamp_frame):
        if self.is_video_loaded is False or self.video_cap is None:
            raise Exception('Video is not loaded yet')
        self.fps = self.video_cap.get(cv2.CAP_PROP_FPS)
        framenumber = int(timestamp_frame)
        self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, framenumber)
        success, frame = self.video_cap.read()

        if success:
            return frame
        else:
            return None