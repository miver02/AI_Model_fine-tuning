from transformers import DistilBertTokenizer, DistilBertModel
from torch.utils.data import Dataset
import torch
import torch.nn as nn
from transformers import Trainer, TrainingArguments
import json
from sklearn.model_selection import train_test_split
import os
from transformers import AdamW
from torch.utils.data import DataLoader

# 自定义数据集类
class ModelNameDataset(Dataset):
    def __init__(self, tokenizer, data_path, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # 加载训练数据
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        input_text = item['input']
        target_text = item['output']
        
        # 编码输入文本
        input_encoding = self.tokenizer(
            input_text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # 编码目标文本
        target_encoding = self.tokenizer(
            target_text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': input_encoding['input_ids'].squeeze(),
            'input_mask': input_encoding['attention_mask'].squeeze(),
            'target_ids': target_encoding['input_ids'].squeeze(),
            'target_mask': target_encoding['attention_mask'].squeeze(),
        }

# 自定义模型类
class ModelNameMatcher(nn.Module):
    def __init__(self, model_name='distilbert-base-uncased', dir='./dataset/model/distilbert-base-uncased'):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained(model_name, cache_dir=dir)
        
    def encode_text(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        # 使用[CLS]标记的输出作为文本表示
        return outputs.last_hidden_state[:, 0, :]
    
    def forward(self, input_ids, input_mask, target_ids=None, target_mask=None):
        # 编码输入文本
        input_embed = self.encode_text(input_ids, input_mask)
        
        if target_ids is not None:
            # 训练模式：计算输入和目标之间的相似度
            target_embed = self.encode_text(target_ids, target_mask)
            # 计算余弦相似度损失
            cos_sim = nn.CosineSimilarity(dim=1)
            loss = 1 - cos_sim(input_embed, target_embed)
            return {'loss': loss.mean()}
        
        return {'embeddings': input_embed}
    
    def find_most_similar(self, input_text, candidate_names, tokenizer):
        # 将输入文本转换为向量
        inputs = tokenizer(
            input_text,
            padding='max_length',
            truncation=True,
            max_length=128,
            return_tensors='pt'
        )
        
        with torch.no_grad():
            input_embed = self.encode_text(inputs['input_ids'], inputs['attention_mask'])
            
            # 计算与所有候选名称的相似度
            max_sim = -1
            most_similar = None
            
            for name in candidate_names:
                name_inputs = tokenizer(
                    name,
                    padding='max_length',
                    truncation=True,
                    max_length=128,
                    return_tensors='pt'
                )
                name_embed = self.encode_text(name_inputs['input_ids'], 
                                            name_inputs['attention_mask'])
                
                similarity = nn.CosineSimilarity(dim=1)(input_embed, name_embed).item()
                
                if similarity > max_sim:
                    max_sim = similarity
                    most_similar = name
                    
            return most_similar

def train_model(train_data_path, model_info_path, save_path, epochs=5):
    # 初始化tokenizer和模型
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', cache_dir='./dataset/model/distilbert-base-uncased')
    model = ModelNameMatcher()
    
    # 准备数据集
    dataset = ModelNameDataset(tokenizer, train_data_path)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)
    
    # 优化器
    optimizer = AdamW(model.parameters(), lr=2e-5)
    
    # 训练循环
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        total_train_loss = 0
        
        for batch in train_loader:
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            input_mask = batch['input_mask'].to(device)
            target_ids = batch['target_ids'].to(device)
            target_mask = batch['target_mask'].to(device)
            
            outputs = model(input_ids, input_mask, target_ids, target_mask)
            loss = outputs['loss']
            
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        
        # 验证阶段
        model.eval()
        total_val_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                input_mask = batch['input_mask'].to(device)
                target_ids = batch['target_ids'].to(device)
                target_mask = batch['target_mask'].to(device)
                
                outputs = model(input_ids, input_mask, target_ids, target_mask)
                loss = outputs['loss']
                
                total_val_loss += loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        
        print(f'Epoch {epoch+1}/{epochs}:')
        print(f'  Training Loss: {avg_train_loss:.4f}')
        print(f'  Validation Loss: {avg_val_loss:.4f}')
        
        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), save_path)
            print(f'  Saved new best model with validation loss: {best_val_loss:.4f}')
        
        print('---')
    
    return model, tokenizer

# 使用示例
def main():
    # 训练模型
    model, tokenizer = train_model(
        train_data_path='./dataset/train_data/train1.json',
        model_info_path='./dataset/train_data/model_info.json',
        save_path='./dataset/save_model/distilbert-miner-model'
    )
    
    # 加载候选模型名称列表
    with open('./dataset/train_data/model_info.json', 'r') as f:
        candidate_names = json.load(f)
    
    # 使用示例
    input_name = "bert base uncased"
    most_similar = model.find_most_similar(input_name, candidate_names, tokenizer)
    print(f"Input: {input_name}")
    print(f"Most similar model name: {most_similar}")

if __name__ == "__main__":
    main()