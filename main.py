from transformers import DistilBertTokenizer
import torch
import json
from train import SimilarityModel

def predict_single(text, model, tokenizer, threshold=0.5):
    """
    预测函数：
    - 输入：任意文本
    - 输出：从model_info中选择最相似的文本（如果相似度超过阈值）
    """
    # 对输入文本进行编码
    inputs = tokenizer(text, padding='max_length', truncation=True, 
                      return_tensors='pt', max_length=128)
    
    # 获取模型预测
    with torch.no_grad():
        outputs = model(inputs['input_ids'], inputs['attention_mask'])
        similarities = outputs['similarities']
        
    # 获取最大相似度及其索引
    best_match_score = similarities[0].max().item()
    best_match_idx = similarities[0].argmax().item()
    
    # 打印所有候选项的相似度（用于调试）
    # for idx, (info, score) in enumerate(zip(model.model_info, similarities[0])):
    #     print(f"候选 {idx}: {info}... 相似度: {score:.4f}")
    
    # 只有当相似度超过阈值时才返回匹配结果
    if best_match_score >= threshold:
        return model.model_info[best_match_idx], best_match_score
    else:
        return "未找到足够相似的匹配", best_match_score

def main():
    # 加载模型和tokenizer
    model_path = './dataset/save_model/distilbert-miner-model'
    tokenizer = DistilBertTokenizer.from_pretrained(model_path)
    model = SimilarityModel.load_model(model_path, tokenizer)
    model.eval()
    
    # 加载测试数据
    test_data = json.load(open('./dataset/train_data/train_test1.json', 'r', encoding='utf-8'))
    
    print("\n开始测试...")
    print("-" * 50)
    print(f"model_info 包含 {len(model.model_info)} 个标准答案")
    print("-" * 50)
    
    for i, item in enumerate(test_data):
        # 获取输入文本
        input_text = item['input'] if isinstance(item, dict) else item
        
        # 从model_info中预测最匹配的答案
        predicted_text, confidence = predict_single(input_text, model, tokenizer)
        
        print(f"\n测试样例 {i+1}:")
        print(f"输入文本: {input_text}")
        print(f"预测答案: {predicted_text}")
        print(f"相似度得分: {confidence:.4f}")
        print("-" * 50)
        
        # 如果是字典格式，比较预测结果和期望输出
        if isinstance(item, dict):
            is_correct = predicted_text == item['output']
            print(f"期望输出: {item['output']}")
            print(f"预测正确: {'✓' if is_correct else '✗'}")
        print("-" * 50)

if __name__ == "__main__":
    main()