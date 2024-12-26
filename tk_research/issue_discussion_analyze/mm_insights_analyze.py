from openai import OpenAI
import json
from tqdm import tqdm
# Initialize the OpenAI client
client = OpenAI(
    base_url="https://api.deepseek.com/",
    api_key="sk-f66d867008a041b7b0521583c3c0ae27"
)


discussions_dir = 'tk_research/issue_discussion_analyze/repo_data'
analyzed_dir = 'tk_research/issue_discussion_analyze/result'


with open(f'{discussions_dir}/discussions.json', 'r') as f:
    discussions = json.load(f)
    
with open(f'{discussions_dir}/issues.json', 'r') as f:
    issues = json.load(f)

merged_discussions = discussions + issues

target_content = []

for chat_dict in tqdm(merged_discussions):
    response = client.chat.completions.create(
        model="deepseek-chat",  # Use the appropriate model
        messages = [
            {
                "role": "system",
                "content": "你是一个提供关于做市策略见解的助手。你的任务是分析文本并提供改进或优化做市策略的建议。"
            },
            {
                "role": "user",
                "content": f"请分析以下文本，并提供关于做市策略的见解或改进建议，如果没有关于做市策略的见解或改进建议就直接输出'无相关内容':{chat_dict}"
            }
        ]
    )
    reply =  response.choices[0].message.content
    if reply == '无相关内容':
        continue
    else:
        target_content.append({
             "title": chat_dict['title'],
             "mm insights": reply
        })
with open(f'{analyzed_dir}/mm_insights.json', 'w', encoding="utf-8") as f:
    json.dump(target_content, f, ensure_ascii=False, indent=4)