# Please install OpenAI SDK first: `pip3 install openai`
import os
from openai import OpenAI
print("start")
client = OpenAI(
    api_key='sk-c2585ec5d417455ab19e063c68979161',
    base_url="https://api.deepseek.com")

response = client.chat.completions.create(
    model="deepseek-reasoner",
    messages=[
        {"role": "system", "content": "你是一个ppt制作专家"},
        {"role": "user", "content": '''按照以下各点生成ppt大纲，注意这个内容是要给企业和政府看的
1.封面：我们是来自xxxx
2.痛点（别人的app的截图）AI请你推荐几个app，分析其优缺点
1.知识库匮乏
2.。。。
3.
3.APP总体介绍（视频+架构）：我们使用鸿蒙5.0版本+华为云服务+Qwen2.5大模型实现了智能编程辅助app，该app能够针对编程初期人员提供xxxx作用
4.功能详细介绍（一个功能一页）
1．用户中心（登录）：使用了鸿蒙xxx接口（哪些技术），用户有学习积分，可以兑换学习资源
2．学习章节（学习树）：使用xxxx技术，ai大模型生成题目
3．对话（联网）：大模型
4．编译器：使用xxx，由于配置环境很麻烦，能够给用户提供编程环境
5.市场
1．高校需求
2．少儿辅助编程
6．社会价值
	1.国家战略需求
	2.农村等资源匮乏的区域，提供数字化xxxx
'''},
    ],
    stream=False,
    temperature=1.7,

)

print(response.choices[0].message.content)
print("faf")