import json
import time

start_time = time.time()

narration_json = json.load(open('data/ego4d_data/v2/annotations/narration.json', 'r'))
print("--- %s seconds ---" % (time.time() - start_time))

print('done')
