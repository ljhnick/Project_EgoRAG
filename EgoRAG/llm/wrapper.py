
import json
from openai import OpenAI
from tqdm import tqdm
import base64
import io
from PIL import Image

def convert_to_base64(image_array):
    img = Image.fromarray(image_array)
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG")
    buffer.seek(0)

    img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return img_base64

class GPTWrapper():

    system_prompt_path = '/home/nick/Research/Project_EgoRAG/llm/prompts/system_prompt_v2.txt'
    user_prompt_path_v2 = '/home/nick/Research/Project_EgoRAG/llm/prompts/user_prompt_v2.txt'
    user_prompt_path = '/home/nick/Research/Project_EgoRAG/llm/prompts/user_prompt.txt'

    def __init__(self):
        self.client = OpenAI()
        self.system_prompt = open(self.system_prompt_path, 'r', encoding='utf-8').read()
        self.user_prompt = open(self.user_prompt_path_v2, 'r', encoding='utf-8').read()
        self.user_prompt_text_only = open(self.user_prompt_path, 'r', encoding='utf-8').read()
        self.parse_user_prompt()

    def parse_user_prompt(self):
        self.user_query = self.user_prompt.split('[PART_ONE]')[0]
        self.user_memory = self.user_prompt.split('[PART_ONE]')[1].split('[PART_TWO]')[0]
        self.user_end = self.user_prompt.split('[PART_TWO]')[1]

    def generate_prompt(self, query, multimodal_input, clip_start_frame, clip_end_frame, text_only=False, spatial_text=False):
        if text_only:
            query_prompt = query
            user_prompt = self.user_prompt_text_only

            user_prompt = user_prompt.replace('[QUERIES]', query_prompt)
            caption_prompt = ""
            for idx, caption in enumerate(multimodal_input):
                timestamp = caption['timestamp_frame']
                start_timestamp = caption['start_frame']
                end_timestamp = caption['end_frame']
                
                caption_text = caption['caption'].replace('#C ', '')
                caption_text = f"{start_timestamp}-{end_timestamp}\t{caption_text}\n"
                caption_prompt += caption_text

            user_prompt = user_prompt.replace('[MEMORIES]', caption_prompt)
            
            return user_prompt
            
        if spatial_text:
            query_prompt = query
            user_prompt = self.user_prompt

            user_prompt = user_prompt.replace('[QUERIES]', query_prompt)
            caption_prompt = ""
            for idx, caption in enumerate(multimodal_input):
                timestamp = caption['timestamp_frame']
                start_timestamp = caption['start_frame']
                end_timestamp = caption['end_frame']
                
                caption_text = caption['caption'].replace('#C ', '')
                spatial_text = caption['spatial_description']

                caption_text = f" {start_timestamp}-{end_timestamp}\t{caption_text}\t{spatial_text}\n"

                caption_prompt += caption_text
            user_prompt = user_prompt.replace('[MEMORIES]', caption_prompt)

            return user_prompt
        else:
            query_prompt = query

            prompt_content = []
            prompt_query = self.user_query.replace('[QUERIES]', query_prompt)
            prompt_query = {"type": "text", "text": prompt_query}

            prompt_memory = {"type": "text", "text": self.user_memory}

            prompt_content.append(prompt_query)
            prompt_content.append(prompt_memory)
            for idx, caption in enumerate(multimodal_input):
                timestamp = caption['timestamp_frame']
                start_timestamp = caption['start_frame']
                end_timestamp = caption['end_frame']
                
                caption_text = caption['caption'].replace('#C ', '')

                caption_text = f"{start_timestamp}-{end_timestamp}\t{caption_text}\n "
                # if timestamp < clip_start_frame or timestamp > clip_end_frame:
                #     continue

                # caption_text = caption['narration_text'].replace('#C ', '')
                # caption_text = f"{timestamp}\t{caption_text}\n"

                caption_entry = {"type": "text", "text": caption_text}
                prompt_content.append(caption_entry)

                if caption['keyframe'] is not None:
                    image_entry = {"type": "image", "image": convert_to_base64(caption['keyframe'])}
                    prompt_content.append(image_entry)
            
            prompt_end = {"type": "text", "text": self.user_end}
            prompt_content.append(prompt_end)

            return prompt_content

    def call_gpt_api_text_only(self, user_prompt):
        is_success = False
        try:
            response = self.client.chat.completions.create(
                model='gpt-4-1106-preview',
                messages=[
                    {'role': "system", "content": self.system_prompt},
                    {'role': "user", "content": user_prompt}
                ],
                temperature=0
            )
            is_success = True
        except Exception as e:
            print(e)
            response = None

        return is_success, response

    def call_gpt_api_text_and_image(self, prompt_pair):
        is_success = False
        
        try:
            response = self.client.chat.completions.create(
                model='gpt-4-vision-preview',
                messages=[
                    {'role': "system", "content": self.system_prompt},
                    {'role': "user", "content": prompt_pair}
                ],
                temperature=0,
                max_tokens=1000
            )
            is_success = True
        except Exception as e:
            print(e)
            response = None

        return is_success, response
    
    def generate_spatial_description(self, keyframe):
        is_success = False

        try:
            response = self.client.chat.completions.create(
                model='gpt-4-vision-preview',
                messages=[
                    {'role': "system", "content": self.system_prompt},
                    {'role': "user", "content": [
                        {"type": "text", "text": "Generate a short dscription of the spatial information in the image."},
                        {"type": "image", "image": convert_to_base64(keyframe)}
                    ]}
                ],
                temperature=0,
                max_tokens=1000
            )
            is_success = True
        except Exception as e:
            print(e)
            response = None
        
        return is_success, response