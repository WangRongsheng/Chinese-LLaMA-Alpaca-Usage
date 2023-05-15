'''
功能：导入必要的第三方库
'''
import openai
import json
import sys
import random
'''
功能：定义输入的OpenAI账号API-key
'''
openai.api_key = ""   # *必填，OpenAI的API-key
if not openai.api_key:
  raise ValueError("OpenAI API key not provided. Please set the 'openai.api_key' variable.")
'''
功能：生成的数据集的prompt，这里指定的生成的数据是通用情况下的，包含多个领域
'''
def return_random_prompt():
  system_prompt = "你需要尽可能给出多样化的任务指令和对应的回答。我们将用于人工评估ChatGPT模型对指令的完成情况。要求:\n"

  # generate random topics
  topic_list = ["科技", "娱乐", "体育", "金融", "时政", "教育", "医疗", "旅游", "美食", "汽车", "房产", "文化", "历史", "地理", "自然", "人文", "社会", "法律", "军事", "政治", "经济", "文学", "艺术", "宗教", "哲学", "语言", "数学", "物理", "化学", "生物", "地球科学", "天文学", "计算机科学", "工程", "建筑", "设计", "音乐", "舞蹈", "电影", "电视", "动漫", "游戏", "健康", "美容", "时尚", "家居", "家电", "家具", "家装", "母婴", "育儿", "职场", "工作", "生活", "养生", "心理", "情感", "人际", "社交", "交友", "恋爱", "婚姻", "家庭", "亲子", "宠物", "动物", "植物", "食品", "饮料", "餐饮", "酒店", "购物", "消费", "理财", "税务", "法规", "法院", "司法", "刑事", "民事", "行政", "战争"]
  system_prompt += "1. 主题多样化，涵盖各个领域，例如：" + "、".join(random.sample(topic_list, 10)) + "等。\n"

  # generate random tasks
  task_list = ["开放式生成", "分类", "问答", "编辑", "摘要", "写作", "翻译", "写代码", "分析", "代码解析", "常识推理", "写信", "抽取", "推荐"]
  system_prompt += "2. 表述多样化，结合真实问题；指令类型多样化，例如：" + "、".join(random.sample(task_list, 10)) + "等。\n"

  # other requirements
  system_prompt += "3. 如果遇到无法处理的指令（只靠文本无法回答），给出无法处理的回复。\n"
  system_prompt += "4. 除非特别要求，请使用中文，指令可以是命令句、疑问句、或其他合适的类型。\n"
  system_prompt += "5. 为指令生成一个适当且涉及真实情况的<input>，不应该只包含简单的占位符。<input>应提供实质性的内容，具有挑战性。字数不超过" + str(random.randint(80, 120)) + "字。\n"
  system_prompt += "6. <output>应该是对指令的适当且真实的回应，不能只回复答应或拒绝请求。如果需要额外信息才能回复时，请努力预测用户意图并尝试回复。<output>的内容应少于" + str(random.randint(128, 512)) + "字。\n\n"
  
  system_prompt += "请给出满足条件的20条JSON格式数据：\n" # *可选，这里的20条，可以改成更多，比如：100、200...

  return system_prompt
'''
功能：主函数
使用：python crawl_prompt.py output.txt |OR| python scripts/crawl_prompt.py output.txt
'''
if __name__ == "__main__":
  if len(sys.argv) != 2:
    print("Usage: python crawl_prompt.py <output_file>")
    exit(1)
  
  output_file = open(sys.argv[1], 'w')

  MAX_EPOCHS = 1    # *必填，要生成的数据数量（每个提示包含20个JSON格式的数据）
  for k in range(MAX_EPOCHS):
    response = openai.ChatCompletion.create(
      model="gpt-3.5-turbo",    # *必填，这里我们使用`gpt-3.5-turbo`模型，而Stanford-Alpaca使用`text-davinci-003`
      messages=[
          {"role": "user", "content": return_random_prompt()},
      ]
    )
    output_file.write(response["choices"][0]["message"]["content"] + '\n')
  output_file.close()
