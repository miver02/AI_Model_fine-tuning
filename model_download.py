from transformers import BartTokenizer, BartForConditionalGeneration

# 指定模型的名称
model_name = 'bart-large'
dir = './dataset/model'  

# 加载 BART 的 tokenizer
tokenizer = BartTokenizer.from_pretrained(model_name, cache_dir=dir)

# 加载 BART 模型
model = BartForConditionalGeneration.from_pretrained(model_name, cache_dir=dir)

# 查看模型和 tokenizer
print(model)
print(tokenizer)