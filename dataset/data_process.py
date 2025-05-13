import json

with open("./PsyQA_full_origin.json", "r", encoding="utf-8") as file:
    data = json.load(file)


# 格式转换
def convert_format(item):
    instruction = (
        f"类型 问题描述#{item['description']}*关键词#"
        f"{'，'.join(item['keywords'].split(',') if isinstance(item['keywords'], str) else item['keywords'])}"
    )
    output = item["answers"][0]["answer_text"].replace(",", "，").strip()
    return {"instruction": instruction, "input": "", "output": output}


converted_data = [convert_format(item) for item in data]

with open("./PsyQA.json", "w", encoding="utf-8") as file:
    json.dump(converted_data, file, ensure_ascii=False, indent=4)
