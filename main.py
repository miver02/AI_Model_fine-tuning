import torch
from transformers import BartTokenizer, BartForConditionalGeneration
import json
from nltk.translate.bleu_score import sentence_bleu

# 指定模型和 tokenizer 的保存路径
model_path = './dataset/save_model/bart_summary_model'

# 加载 tokenizer 和模型
tokenizer = BartTokenizer.from_pretrained(model_path)
model = BartForConditionalGeneration.from_pretrained(model_path)

def load_test_data(test_data_path, tokenizer, max_len=512, test_limit=1000):
    with open(test_data_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    test_encodings = [tokenizer(article['input'], return_tensors="pt", max_length=max_len, truncation=True, padding="max_length") for article in test_data[:test_limit]]
    return test_data, test_encodings

def evaluate_model(model, tokenizer, test_data, test_encodings, device):
    model.to(device)
    model.eval()
    scores = []
    print("\n" + "="*80 + "\n")  # 分隔线
    
    for i, (article, encoding) in enumerate(zip(test_data, test_encodings)):
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        
        # 生成摘要
        summary_ids = model.generate(input_ids, attention_mask=attention_mask, num_beams=4, max_length=100, early_stopping=True)
        generated_summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        
        # 计算 BLEU 分数
        reference = article['output']
        score = sentence_bleu([reference.split()], generated_summary.split())
        scores.append(score)
        
        # 打印详细信息
        print(f"样本 {i+1}:")
        print(f"原文: {article['input']}")
        print(f"摘要: {generated_summary}")
        print(f"参考: {reference}")
        print(f"BLEU分数: {score:.4f}")
        print("\n" + "-"*80 + "\n")  # 分隔线
    
    # 计算并返回平均 BLEU 分数
    average_score = sum(scores) / len(scores)
    print(f"总体评估:")
    print(f"样本数量: {len(scores)}")
    print(f"平均 BLEU 分数: {average_score:.4f}")
    return average_score

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    test_data_path = './dataset/train_data/train.json'
    print(f"加载测试数据: {test_data_path}")
    
    test_data, test_encodings = load_test_data(test_data_path, tokenizer)
    print(f"测试样本数量: {len(test_data)}")
    
    average_score = evaluate_model(model, tokenizer, test_data, test_encodings, device)

if __name__ == "__main__":
    main()