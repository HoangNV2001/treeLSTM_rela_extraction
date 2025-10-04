import torch
import json
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.metrics import classification_report, f1_score
import argparse

from data_processing import DependencyParser, RelationDataset, collate_fn
from model import RelationExtractionModel

def test(model, dataloader, id2label, device, output_file='predictions.jsonl'):
    model.eval()
    all_preds = []
    all_sent_ids = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            metadata = batch['metadata']
            
            logits = model(input_ids, attention_mask, metadata)
            preds = torch.argmax(logits, dim=-1)
            
            all_preds.extend(preds.cpu().numpy())
            all_sent_ids.extend([m['sent_id'] for m in metadata])
    
    # Write predictions
    with open(output_file, 'w', encoding='utf-8') as f:
        for sent_id, pred_id in zip(all_sent_ids, all_preds):
            pred_label = id2label[pred_id]
            result = {
                'sent_id': sent_id,
                'pred': pred_label
            }
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    print(f"Predictions saved to {output_file}")
    return all_preds

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_file', type=str, default='test.jsonl')
    parser.add_argument('--model_path', type=str, default='best_model.pt')
    parser.add_argument('--vncorenlp_path', type=str, default='VnCoreNLP/VnCoreNLP-1.1.1.jar')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--output_file', type=str, default='predictions.jsonl')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model checkpoint
    checkpoint = torch.load(args.model_path, map_location=device)
    label2id = checkpoint['label2id']
    id2label = checkpoint['id2label']
    num_labels = len(label2id)
    
    # Initialize parser and tokenizer
    print("Initializing vnCoreNLP...")
    parser_vncore = DependencyParser(args.vncorenlp_path)
    tokenizer = AutoTokenizer.from_pretrained('vinai/phobert-base-v2')
    
    # Load test data
    print("Loading test data...")
    test_dataset = RelationDataset(args.test_file, parser_vncore, tokenizer, is_test=True)
    test_dataset.label2id = label2id
    test_dataset.id2label = id2label
    
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, 
                            shuffle=False, collate_fn=collate_fn)
    
    # Initialize model
    model = RelationExtractionModel(num_labels=num_labels).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Test
    print("Running inference...")
    predictions = test(model, test_loader, id2label, device, args.output_file)
    
    # If test file has labels, compute metrics
    has_labels = 'label' in test_dataset.data[0]
    if has_labels:
        gold_labels = [label2id.get(item.get('label', 'Other'), 0) for item in test_dataset.data]
        
        print("\nTest Results:")
        print(classification_report([id2label[g] for g in gold_labels], 
                                   [id2label[p] for p in predictions]))
        
        f1 = f1_score(gold_labels, predictions, average='macro')
        print(f"\nMacro F1: {f1:.4f}")

if __name__ == '__main__':
    main()