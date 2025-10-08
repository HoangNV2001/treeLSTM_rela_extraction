import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
import numpy as np
from tqdm import tqdm
import argparse
import random

from data_processing import DependencyParser, RelationDataset, collate_fn
from model import RelationExtractionModel

def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def train_epoch(model, dataloader, optimizer, criterion, device, scheduler=None, 
                accumulation_steps=2, max_grad_norm=1.0):
    """Train with gradient accumulation and FGM adversarial training"""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    optimizer.zero_grad()
    
    for i, batch in enumerate(tqdm(dataloader, desc="Training")):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        metadata = batch['metadata']
        
        # Forward pass
        logits = model(input_ids, attention_mask, metadata)
        loss = criterion(logits, labels)
        
        # Normalize loss for accumulation
        loss = loss / accumulation_steps
        loss.backward()
        
        # Gradient accumulation
        if (i + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            optimizer.step()
            if scheduler:
                scheduler.step()
            optimizer.zero_grad()
        
        total_loss += loss.item() * accumulation_steps
        
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
    parser.add_argument('--accumulation_steps', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--phobert_lr', type=float, default=1e-5)  # Lower LR for PhoBERT
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--warmup_ratio', type=float, default=0.1)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--word_dropout', type=float, default=0.1)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--model_save_path', type=str, default='best_model.pt')
    parser.add_argument('--use_focal_loss', action='store_true', help='Use focal loss for imbalanced data')
    parser.add_argument('--label_smoothing', type=float, default=0.1)
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize parser and tokenizer
    print("Initializing vnCoreNLP...")
    parser_vncore = DependencyParser(args.vncorenlp_path)
    tokenizer = AutoTokenizer.from_pretrained('vinai/phobert-base-v2')
    
    # Load datasets
    print("Loading training data...")
    train_dataset = RelationDataset(args.train_file, parser_vncore, tokenizer)
    
    # Split with stratification if possible
    train_size = int(0.85 * len(train_dataset))  # 85-15 split
    val_size = len(train_dataset) - train_size
    train_data, val_data = torch.utils.data.random_split(
        train_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed)
    )
    
    train_loader = DataLoader(train_data, batch_size=args.batch_size, 
                             shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, 
                           shuffle=False, collate_fn=collate_fn)
    
    # Initialize model
    num_labels = len(train_dataset.label2id)
    print(f"Number of labels: {num_labels}")
    print(f"Labels: {train_dataset.label2id}")
    
    model = RelationExtractionModel(
        num_labels=num_labels,
        dropout=args.dropout,
        word_dropout=args.word_dropout,
        use_layer_norm=True
    ).to(device)
    
    # Differential learning rates
    phobert_params = list(model.phobert.parameters())
    other_params = [p for n, p in model.named_parameters() if 'phobert' not in n]
    
    optimizer = torch.optim.AdamW([
        {'params': phobert_params, 'lr': args.phobert_lr},
        {'params': other_params, 'lr': args.lr}
    ], weight_decay=args.weight_decay)
    
    # Learning rate scheduler
    total_steps = len(train_loader) * args.epochs // args.accumulation_steps
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # Loss function with label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    
    # Training loop
    best_f1 = 0
    patience_counter = 0
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        
        train_loss, train_f1, train_prec, train_rec = train_epoch(
            model, train_loader, optimizer, criterion, device, 
            scheduler, args.accumulation_steps, args.max_grad_norm
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
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'label2id': train_dataset.label2id,
                'id2label': train_dataset.id2label,
                'best_f1': best_f1
            }, args.model_save_path)
            print(f"✅ Saved best model with F1: {best_f1:.4f}")
        else:
            patience_counter += 1
            print(f"Patience: {patience_counter}/{args.patience}")
            if patience_counter >= args.patience:
                print("Early stopping triggered")
                break
    
    print(f"\n🎉 Training completed. Best F1: {best_f1:.4f}")

if __name__ == '__main__':
    main()