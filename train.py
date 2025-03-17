from transformers import DistilBertTokenizer, DistilBertModel
from torch.utils.data import Dataset
import torch
import torch.nn as nn
from transformers import Trainer, TrainingArguments
import json
from sklearn.model_selection import train_test_split
import os

# 自定义数据集类
class MinerDataset(Dataset):
    def __init__(self, tokenizer, train_data, model_info):
        self.tokenizer = tokenizer
        self.train_data = train_data
        self.model_info = model_info
        
        # 创建输出到索引的映射
        self.output_to_idx = {str(info): idx for idx, info in enumerate(model_info)}
        
    def __len__(self):
        return len(self.train_data)
    
    def __getitem__(self, idx):
        item = self.train_data[idx]
        # 对输入文本进行编码
        inputs = self.tokenizer(item['input'], padding='max_length', truncation=True, 
                              return_tensors='pt', max_length=128)
        
        # 找到目标输出在 model_info 中的索引
        target_idx = self.output_to_idx.get(str(item['output']))
        if target_idx is None:
            # 如果输出不在 model_info 中，找到最相似的
            target_encoding = self.tokenizer(str(item['output']), padding='max_length', 
                                           truncation=True, return_tensors='pt', max_length=128)
            # 使用原始输出
            return {
                'input_ids': inputs['input_ids'].squeeze(),
                'attention_mask': inputs['attention_mask'].squeeze(),
                'target_ids': target_encoding['input_ids'].squeeze(),
                'target_mask': target_encoding['attention_mask'].squeeze(),
            }
        
        # 使用 model_info 中的标准答案
        target_text = self.model_info[target_idx]
        target_encoding = self.tokenizer(str(target_text), padding='max_length', 
                                       truncation=True, return_tensors='pt', max_length=128)
        
        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'target_ids': target_encoding['input_ids'].squeeze(),
            'target_mask': target_encoding['attention_mask'].squeeze(),
        }

# 自定义模型类
class SimilarityModel(nn.Module):
    def __init__(self, model_name, model_info, tokenizer):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained(model_name)
        self.tokenizer = tokenizer
        self.model_info = model_info
        
        # 预计算所有model_info的编码
        self.info_encodings = []
        with torch.no_grad():
            for info in model_info:
                inputs = tokenizer(str(info), padding='max_length', truncation=True, 
                                 return_tensors='pt', max_length=128)
                encoding = self.bert(**inputs).last_hidden_state.mean(dim=1)
                # 归一化编码向量
                encoding = torch.nn.functional.normalize(encoding, p=2, dim=1)
                self.info_encodings.append(encoding)
        self.info_encodings = torch.cat(self.info_encodings, dim=0)
        
    def forward(self, input_ids, attention_mask, target_ids=None, target_mask=None):
        # 获取输入文本的编码
        input_encoding = self.bert(input_ids, attention_mask=attention_mask).last_hidden_state.mean(dim=1)
        # 归一化输入编码
        input_encoding = torch.nn.functional.normalize(input_encoding, p=2, dim=1)
        
        # 计算余弦相似度
        similarities = torch.matmul(input_encoding, self.info_encodings.T)
        # similarities 现在的范围是 [-1, 1]，转换到 [0, 1]
        similarities = (similarities + 1) / 2
        
        if target_ids is not None:
            target_encoding = self.bert(target_ids, attention_mask=target_mask).last_hidden_state.mean(dim=1)
            target_encoding = torch.nn.functional.normalize(target_encoding, p=2, dim=1)
            loss = 1 - torch.cosine_similarity(input_encoding, target_encoding)
            return {'loss': loss.mean(), 'similarities': similarities}
        
        return {'similarities': similarities}

    def save_model(self, save_path):
        """保存模型和相关数据"""
        os.makedirs(save_path, exist_ok=True)
        # 保存 BERT 模型
        self.bert.save_pretrained(save_path)
        # 保存 model_info
        with open(os.path.join(save_path, 'model_info.json'), 'w', encoding='utf-8') as f:
            json.dump(self.model_info, f, ensure_ascii=False, indent=2)
        # 保存 info_encodings
        torch.save(self.info_encodings, os.path.join(save_path, 'info_encodings.pt'))

    @classmethod
    def load_model(cls, load_path, tokenizer):
        """加载保存的模型"""
        # 加载 model_info
        with open(os.path.join(load_path, 'model_info.json'), 'r', encoding='utf-8') as f:
            model_info = json.load(f)
        # 创建模型实例
        model = cls(load_path, model_info, tokenizer)
        # 加载保存的 info_encodings
        model.info_encodings = torch.load(os.path.join(load_path, 'info_encodings.pt'))
        return model

# 主要训练代码
def main():
    # 加载数据
    model_info = json.load(open('./dataset/train_data/model_info.json', 'r', encoding='utf-8'))
    train_data = json.load(open('./dataset/train_data/train1.json', 'r', encoding='utf-8'))
    
    # 分割训练集和验证集
    train_data, eval_data = train_test_split(train_data, test_size=0.2, random_state=42)
    
    # 初始化tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', 
                                                   cache_dir='./dataset/model/distilbert-base-uncased')
    
    # 创建训练集和验证集
    train_dataset = MinerDataset(tokenizer, train_data, model_info)
    eval_dataset = MinerDataset(tokenizer, eval_data, model_info)
    
    # 创建模型
    model = SimilarityModel('distilbert-base-uncased', model_info, tokenizer)
    
    # 训练参数
    training_args = TrainingArguments(
        output_dir='./dataset/results',          # 模型训练过程中的输出目录，保存检查点和中间结果
        num_train_epochs=5,              # 训练轮数，表示整个数据集要被训练5遍
        per_device_train_batch_size=20,  # 每个批次的样本数量，这里每次处理16个样本
        logging_dir='./dataset/logs',            # 训练日志保存的目录
        logging_steps=10,                # 每10步记录一次训练状态
        save_total_limit=5,              # 最多保存2个检查点，旧的会被新的覆盖以节省空间
    )
    
    # 定义Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    
    # 开始训练
    trainer.train()
    
    # 保存模型
    save_path = './dataset/save_model/distilbert-miner-model'
    model.save_model(save_path)
    tokenizer.save_pretrained(save_path)

if __name__ == "__main__":
    main()