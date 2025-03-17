from transformers import DistilBertTokenizer, DistilBertModel

# 加载 DistilBERT 的 tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', cache_dir='./dataset/model/distilbert-base-uncased')

# 加载 DistilBERT 模型
model = DistilBertModel.from_pretrained('distilbert-base-uncased', cache_dir='./dataset/model/distilbert-base-uncased')

# 查看模型和 tokenizer
print(model)
print(tokenizer)