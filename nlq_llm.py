import json
from openai import OpenAI
from tqdm import tqdm

client = OpenAI()
def gpt_completion_api(system_prompt, user_prompt):
    response = client.chat.completions.create(
        model='gpt-4-1106-preview',
        messages=[
            {'role': "system", "content": system_prompt},
            {'role': "user", "content": user_prompt}
        ],
        temperature=0
    )

    return response

def parse_prediction(prediction):
    element = {}
    if prediction == 'NA':
        element['pred_start'] = 0
        element['pred_end'] = 0
        element['NA'] = True
    else:
        element['NA'] = False
        pred = eval(prediction)
        if isinstance(pred, list):
            pred_start = pred[0]
            pred_end = pred[1]
        elif isinstance(pred, tuple):
            pred_start = pred[0]
            pred_end = pred[1]
        else:
            if '-' in prediction:
                pred_start = int(prediction.split('-')[0])
                pred_end = int(prediction.split('-')[1])
            else:
                pred_start = 0
                pred_end = 0
                element['NA'] = True
                print(f'Error: {prediction}')
        
        element['pred_start'] = pred_start
        element['pred_end'] = pred_end
    return element

def process_result(result, output):
    json_result = json.loads(result)
    for res in json_result:
        index = res['query_index']
        
        predictions = res['predictions']
        element = {}

        # check if prediction contains NA
        element = parse_prediction(predictions)
        # if predictions == 'NA':
        #     element['pred_start'] = 0
        #     element['pred_end'] = 0
        #     element['NA'] = True
        # else:
        #     pred = eval(predictions)
        #     element['pred_start'] = pred[0]
        #     element['pred_end'] = pred[1]
        #     element['NA'] = False
            

        output[index] = element

    return output


# generate prompt
def generate_prompt(queries, captions, start_id=0, num=5, system_prompt_path='', user_prompt_path=''):
    clip_frames_init = queries['clip_frames'][start_id]
    start = clip_frames_init[0]
    end = clip_frames_init[1]

    index = start_id # start index
    query_prompt = ""
    for i in range(num):
        if start_id+i >= len(queries['query']):
            break
        clip_frames = queries['clip_frames'][start_id+i]
        end_id = start_id+i
        if clip_frames != clip_frames_init:
            end_id -= 1
            break
        query = queries['query'][start_id+i]
        query_prompt += f"{index+i}\t{query}\n"

    caption_prompt = ""
    for idx, caption_text in enumerate(captions['caption']):
        text = caption_text.replace('#C ', '')
        timestamp = captions['timestamps'][idx]
        if timestamp < start or timestamp > end:
            continue
        caption_prompt += f"{timestamp}\t{text}\n"
    
    # read all lines from system_prompt.txt
    with open(user_prompt_path, 'r') as f:
        user_prompt = f.read()
    
    user_prompt = user_prompt.replace('[QUERIES]', query_prompt)
    user_prompt = user_prompt.replace('[TIMESTAMP_AND_CAPTION]', caption_prompt)

    end_id += 1
    return user_prompt, end_id

def check_keys(anno):
    if 'query' not in anno:
        return False
    if 'template' not in anno:
        return False
    if 'clip_start_frame' not in anno:
        return False
    if 'clip_end_frame' not in anno:
        return False
    if 'video_start_frame' not in anno:
        return False
    if 'video_end_frame' not in anno:
        return False
    return True
        

def main():
    data_processed = json.load(open('data/data_process_val_all.json', 'r'))
    system_prompt_path = 'llm/prompts/system_prompt.txt'
    user_prompt_path = 'llm/prompts/user_prompt.txt'

    system_prompt = open(system_prompt_path, 'r').read()

    error_uids = []

    predicted_results = {}

    video_index = -1
    for video_uid in tqdm(data_processed):
        video_index += 1
        if video_index < 207:
            continue

        data = data_processed[video_uid]
        narration = data['narration']
        annotations = data['annotations']

        # generate narration
        captions = {}
        captions['timestamps'] = []
        captions['caption'] = []
        
        for nar in narration:
            captions['timestamps'].append(nar['timestamp_frame'])
            captions['caption'].append(nar['narration_text'])

        # generate query
        # we concatenate all the language query into same API call
        queries = {}
        query_list = []
        clip_frames = []
        gt = []
        templates = []
        for anno in annotations:
            if not check_keys(anno):
                continue
            query_list.append(anno['query'])
            templates.append(anno['template'])
            frames = (anno['clip_start_frame'], anno['clip_end_frame'])
            clip_frames.append(frames)
            gt_frames = anno['video_start_frame'], anno['video_end_frame']
            gt.append(gt_frames)
        queries['query'] = query_list
        queries['clip_frames'] = clip_frames
        queries['gt'] = gt
        queries['template'] = templates

        start_id = 0
        processed_result = {}
        print('Calling GPT API...')
        is_error = False
        while True:
            try:
                user_prompt, end_id = generate_prompt(queries, captions, start_id=start_id,system_prompt_path=system_prompt_path, user_prompt_path=user_prompt_path, num=10)
            except Exception as e:
                print(e)
                break
            # calling gpt API
            print('start_id: ', start_id, 'end_id: ', end_id)
            response = gpt_completion_api(system_prompt, user_prompt)
            result = response.choices[0].message.content

            try:
                processed_result = process_result(result, processed_result)
            except Exception as e:
                print(e)
                print('video_uid: ', video_uid)
                error_uids.append(video_uid)
                is_error = True
                break

            if end_id == len(queries['query']):
                break
            start_id = end_id

        predicted_results[video_uid] = {}
        predicted_results[video_uid]['result'] = []
        

        if is_error:
            is_error = False
            continue

        for i in range(len(queries['query'])):
            # save query, gt start and end, predicted start and end
            data = {}
            data['query'] = queries['query'][i]
            data['clip_frames'] = queries['clip_frames'][i]
            data['template'] = queries['template'][i]
            data['gt_start'] = queries['gt'][i][0]
            data['gt_end'] = queries['gt'][i][1]
            data['predicted_start'] = processed_result[i]['pred_start']
            data['predicted_end'] = processed_result[i]['pred_end']
            data['NA'] = processed_result[i]['NA']
            predicted_results[video_uid]['result'].append(data)

        print('done')
    
    # save predicted results as json file
        
    print('Error uids: ', error_uids)

    with open('data/predicted_results_207_after.json', 'w') as f:
        json.dump(predicted_results, f)

if __name__ == '__main__':
    main()