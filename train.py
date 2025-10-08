import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
import numpy as np
from tqdm import tqdm
import argparse

from data_processing import DependencyParser, RelationDataset, collate_fn
from model import RelationExtractionModel

def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for batch in tqdm(dataloader, desc="Training"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        metadata = batch['metadata']
        
        optimizer.zero_grad()
        
        logits = model(input_ids, attention_mask, metadata)
        loss = criterion(logits, labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        
        preds = torch.argmax(logits, dim=-1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    f1 = f1_score(all_labels, all_preds, average='macro')
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    
    return avg_loss, f1, precision, recall

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            metadata = batch['metadata']
            
            logits = model(input_ids, attention_mask, metadata)
            loss = criterion(logits, labels)
            
            total_loss += loss.item()
            
            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    f1 = f1_score(all_labels, all_preds, average='macro')
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    
    return avg_loss, f1, precision, recall, all_preds, all_labels

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file', type=str, default='train.jsonl')
    parser.add_argument('--vncorenlp_path', type=str, default='VnCoreNLP/VnCoreNLP-1.1.1.jar')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--model_save_path', type=str, default='best_model.pt')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize parser and tokenizer
    print("Initializing vnCoreNLP...")
    parser_vncore = DependencyParser(args.vncorenlp_path)
    tokenizer = AutoTokenizer.from_pretrained('vinai/phobert-base-v2')
    
    # Load datasets
    print("Loading training data...")
    train_dataset = RelationDataset(args.train_file, parser_vncore, tokenizer)
    
    # Split train/val
    train_size = int(0.9 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_data, val_data = torch.utils.data.random_split(train_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_data, batch_size=args.batch_size, 
                             shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, 
                           shuffle=False, collate_fn=collate_fn)
    
    # Initialize model
    num_labels = len(train_dataset.label2id)
    print(f"Number of labels: {num_labels}")
    print(f"Labels: {train_dataset.label2id}")
    
    model = RelationExtractionModel(num_labels=num_labels).to(device)
    
    # Optimizer and loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    best_f1 = 0
    patience = 3
    patience_counter = 0
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        
        train_loss, train_f1, train_prec, train_rec = train_epoch(
            model, train_loader, optimizer, criterion, device
        )
        
        val_loss, val_f1, val_prec, val_rec, _, _ = evaluate(
            model, val_loader, criterion, device
        )
        
        print(f"Train Loss: {train_loss:.4f}, F1: {train_f1:.4f}, "
              f"Precision: {train_prec:.4f}, Recall: {train_rec:.4f}")
        print(f"Val Loss: {val_loss:.4f}, F1: {val_f1:.4f}, "
              f"Precision: {val_prec:.4f}, Recall: {val_rec:.4f}")
        
        # Early stopping
        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'label2id': train_dataset.label2id,
                'id2label': train_dataset.id2label
            }, args.model_save_path)
            print(f"Saved best model with F1: {best_f1:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break
    
    print(f"\nTraining completed. Best F1: {best_f1:.4f}")

if __name__ == '__main__':
    main()