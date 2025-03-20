import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BartTokenizer, BartForConditionalGeneration
from transformers import AdamW, get_linear_schedule_with_warmup
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SummaryDataset(Dataset):
    def __init__(self, tokenizer, data_path, max_len=512, summary_len=128):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.summary_len = summary_len
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        article = item['input']
        summary = item['output']
        
        article_encoding = self.tokenizer(
            article,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        summary_encoding = self.tokenizer(
            summary,
            max_length=self.summary_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        labels = summary_encoding.input_ids
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            'input_ids': article_encoding.input_ids.flatten(),
            'attention_mask': article_encoding.attention_mask.flatten(),
            'labels': labels.flatten()
        }

def train_model(train_data_path, save_path, epochs=5):
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
    model = BartForConditionalGeneration.from_pretrained('facebook/bart-large')
    dataset = SummaryDataset(tokenizer, train_data_path)
    train_loader = DataLoader(dataset, batch_size=4, shuffle=True)
    optimizer = AdamW(model.parameters(), lr=2e-5)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        logger.info(f'Epoch {epoch+1}/{epochs}: Training Loss: {total_train_loss / len(train_loader):.4f}')

    model_save_path = os.path.join(save_path, 'bart_summary_model')
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    logger.info(f'Model and tokenizer saved to {model_save_path}')

def main():
    train_model(
        train_data_path='./dataset/train_data/train.json',
        save_path='./dataset/save_model'
    )

if __name__ == "__main__":
    main()