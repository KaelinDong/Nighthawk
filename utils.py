import json

def read_jsonl(str_jsonl):
    list_dict_data = []
    with open(str_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            list_dict_data.append(obj)

    return list_dict_data


def write_jsonl(list_dict_data, str_jsonl):
    with open(str_jsonl, "w", encoding="utf-8") as file_jsonl:
        for dict_data in list_dict_data:
            file_jsonl.write(json.dumps(dict_data, ensure_ascii=False) + "\n")