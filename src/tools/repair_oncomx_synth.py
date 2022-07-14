from pathlib import Path
import json

def main():
    dict_file = Path("data/oncomx/generative/syntetic_queries.json")
    data_file = Path("data/oncomx/handmade_training_data/all_synthetic_data_jan_after_critic.json")
    output_file = Path("data/oncomx/handmade_training_data/generative_data_train.json")
    with open(dict_file, 'r') as f:
        dict_data = json.load(f)
        _dict = {}
        for d in dict_data:
            _dict[d['readable_query']] = d['query']
    
    with open(data_file, 'r') as f:
        _data = json.load(f)
    res = []
    for d in _data:
        temp = _dict.get(d['query'], None)
        if temp is not None:
            d['query'] = temp
            res.append(d)
    print(len(res))
    with open(output_file, 'w') as f:
        json.dump(res, f, indent=2)

if __name__ == '__main__':
    main()