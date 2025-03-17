from transformers import DistilBertTokenizer
import torch
import json
from train import ModelNameMatcher

def predict_single(input_text, model, tokenizer, candidate_names):
    """
    预测单个输入文本的最相似模型名称
    返回：预测的模型名称和相似度分数
    """
    # 将模型移动到与输入相同的设备
    device = next(model.parameters()).device
    
    # 将输入文本转换为向量
    inputs = tokenizer(
        input_text,
        padding='max_length',
        truncation=True,
        max_length=128,
        return_tensors='pt'
    )
    
    # 将输入移动到正确的设备
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # 获取预测结果
    most_similar = model.find_most_similar(input_text, candidate_names, tokenizer)
    
    # 计算相似度分数
    with torch.no_grad():
        input_embed = model.encode_text(inputs['input_ids'], inputs['attention_mask'])
        target_inputs = tokenizer(
            most_similar,
            padding='max_length',
            truncation=True,
            max_length=128,
            return_tensors='pt'
        ).to(device)
        target_embed = model.encode_text(target_inputs['input_ids'], 
                                       target_inputs['attention_mask'])
        
        similarity = torch.nn.functional.cosine_similarity(input_embed, target_embed).item()
    
    return most_similar, similarity

def evaluate_model(test_data_path, model_path, model_info_path):
    """评估模型性能"""
    # 修改tokenizer的加载方式
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', 
                                                   cache_dir='./dataset/model/distilbert-base-uncased')
    
    # 加载模型
    model = ModelNameMatcher()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # 如果有GPU则使用GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # 加载测试数据和候选名称列表
    with open(test_data_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    with open(model_info_path, 'r', encoding='utf-8') as f:
        candidate_names = json.load(f)
    
    # 统计信息
    total = len(test_data)
    correct = 0
    
    print("\n开始模型评估...")
    print("-" * 60)
    
    for i, item in enumerate(test_data, 1):
        input_text = item['input']
        expected_output = item['output']
        
        # 获取预测结果
        predicted_output, similarity = predict_single(input_text, model, tokenizer, candidate_names)
        
        # 检查预测是否正确
        is_correct = predicted_output == expected_output
        if is_correct:
            correct += 1
            
        # 打印详细信息
        print(f"测试样例 {i}:")
        print(f"输入: {input_text}")
        print(f"预期输出: {expected_output}")
        print(f"预测输出: {predicted_output}")
        print(f"相似度分数: {similarity:.4f}")
        print(f"预测结果: {'✓ 正确' if is_correct else '✗ 错误'}")
        print("-" * 60)
    
    # 打印总体评估结果
    accuracy = correct / total * 100
    print(f"\n评估结果:")
    print(f"总样本数: {total}")
    print(f"正确预测数: {correct}")
    print(f"准确率: {accuracy:.2f}%")

def main():
    # 设置路径
    test_data_path = './dataset/train_data/train1.json'
    model_path = './dataset/save_model/distilbert-miner-model'
    model_info_path = './dataset/train_data/model_info.json'
    
    # 评估模型
    evaluate_model(test_data_path, model_path, model_info_path)
    
    # 交互式测试部分也需要修改tokenizer的加载方式
    print("\n开始交互式测试 (输入 'q' 退出):")
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', 
                                                   cache_dir='./dataset/model/distilbert-base-uncased')
    model = ModelNameMatcher()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    with open(model_info_path, 'r', encoding='utf-8') as f:
        candidate_names = json.load(f)
    
    while True:
        user_input = input("\n请输入模型名称: ")
        if user_input.lower() == 'q':
            break
            
        predicted_output, similarity = predict_single(user_input, model, tokenizer, candidate_names)
        print(f"预测的标准名称: {predicted_output}")
        print(f"相似度分数: {similarity:.4f}")

if __name__ == "__main__":
    main()