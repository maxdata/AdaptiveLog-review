import argparse
from tqdm import tqdm
import os
import time
import json
import random

prompt = '''
            You are a professional operations engineer and your task is to analyze whether given logs and natural language descriptions are relevant.
            You can refer to the following error-prone cases, learn key features from these cases and attention common pitfalls in prediction.
            Please 1. Describe the reasoning process firstly by referring to the reasoning process of relevant error-prone cases. 
            2. Follow the label format of examples and give a definite result.
            {}

            The following input is the test data.
            Please 1. Describe the reasoning process (e.g. Reason: xxx) firstly by referring to the reasons of relevant error-prone cases. 2. Finally, follow the label format (e.g. Label: xxx) and give a definite result.
            Input: [{}, {}]
            '''

headers = {
        "Authorization": "Bearer sb-$OPENAI_API_KEY",
        "Content-Type": "application/json",
}

def parse_args():
    args = argparse.ArgumentParser()
    # network arguments

    args.add_argument("-data", "--data", type=str,help="test input data")

    args.add_argument("-prompt", "--prompt", type=str, default=prompt,
                      help="prompt of error-case reasoning")

    args.add_argument("-url", "--url", type=str, default='https://api.openai.com/v1/chat/completions' help="ChatGPT URL")

    args.add_argument("-headers", "--headers", type=dict, default=headers, help="tokens")


    args.add_argument("-model", "--model", default= 'gpt-3.5-turbo-16k-0613',type=str, help="model version of ChatGPT")

    args.add_argument("-save_path", "--save_path", type=str, default='output/chatgpt_result.json',help="Folder name to save the result.")

    args = args.parse_args()
    return args


def read_json(file):
    with open(file, 'r+') as file:
        content = file.read()
    content = json.loads(content)
    return content


def save_json(data, file):
    dict_json = json.dumps(data, indent=1)
    with open(file, 'w+', newline='\n') as file:
        file.write(dict_json)


def save_text(data, file):
    with open(file, 'a') as f:
        f.write(data)


def read_text(file):
    c = 0
    with open(file, 'r') as file:
        for line in file:
            if line.strip() != '':
                c += 1
    return c


import requests

def get_data_from_chatGPT2(prompt,data,case,headers,url,model,save_path):
    result = []
    cnt = 1

    data = read_json(data)
    # cc = read_text("result/ldsm_h3csecurity_result_chatgpt.txt")
    cc = 0
    for i in range(cc, len(data)):
        first_question = prompt.format(case, data[i][0][0][0], data[i][0][0][1])
        a = requests.post(url, json={
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": first_question
                }
            ],
            "temperature": 0,
            "stream": False,
            "max_tokens": 4096,
        }, headers=headers)
        content1 = a.json()['choices'][0]['message']['content']
        # print(a.json()['choices'][0]['message']['content'])
        print(i, data[i][0][1], data[i][0][0], content1)
        result.append([str(data[i][0][1]), content1])

        text_result = str(i) + '\t' + str(data[i][0][1]) + '\t' + content1 + "\n"
        # save_text(text_result, "output/ldsm_hwrouters_cbr_chatgpt.txt")
        print('-*' * 20)
        if i == len(data)-1:
            save_json(result,save_path)

    return result



if __name__ == '__main__':
    args = parse_args()
    get_data_from_chatGPT2(args)