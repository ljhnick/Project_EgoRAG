import json

data_1 = json.load(open('data/predicted_results_207.json', 'r'))
data_2 = json.load(open('data/predicted_results_207_after.json', 'r'))

data = data_1.copy()

print(len(data_1))
print(len(data_2))

for key in data_2:
    data[key] = data_2[key]

with open('data/predicted_results.json', 'w') as f:
    json.dump(data, f)