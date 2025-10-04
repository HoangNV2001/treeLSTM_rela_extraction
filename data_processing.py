import json
import numpy as np
from typing import List, Dict, Tuple
from vncorenlp import VnCoreNLP
from transformers import AutoTokenizer
import torch

class DependencyParser:
    def __init__(self, vncorenlp_path="VnCoreNLP/VnCoreNLP-1.1.1.jar"):
        """
        Initialize vnCoreNLP for dependency parsing
        Download from: https://github.com/vncorenlp/VnCoreNLP
        """
        self.annotator = VnCoreNLP(vncorenlp_path, annotators="wseg,pos,parse", max_heap_size='-Xmx2g')
    
    def parse(self, sentence: str) -> Dict:
        """Parse sentence and return tokens, POS tags, and dependency tree"""
        result = self.annotator.annotate(sentence)
        
        tokens = []
        pos_tags = []
        dep_heads = []
        dep_rels = []
        
        for sent in result['sentences']:
            for word in sent:
                tokens.append(word['form'])
                pos_tags.append(word['posTag'])
                dep_heads.append(int(word['head']) - 1)  # Convert to 0-indexed
                dep_rels.append(word['depLabel'])
        
        return {
            'tokens': tokens,
            'pos_tags': pos_tags,
            'dep_heads': dep_heads,
            'dep_rels': dep_rels
        }
    
    def find_shortest_path(self, dep_heads: List[int], e1_idx: int, e2_idx: int) -> List[int]:
        """Find shortest dependency path between two entities"""
        # Build parent dict
        n = len(dep_heads)
        
        # Find path from e1 to root
        path1 = []
        curr = e1_idx
        visited = set()
        while curr != -1 and curr not in visited:
            path1.append(curr)
            visited.add(curr)
            curr = dep_heads[curr]
        
        # Find path from e2 to root
        path2 = []
        curr = e2_idx
        visited2 = set()
        while curr != -1 and curr not in visited2:
            if curr in visited:
                # Found common ancestor
                common_idx = path1.index(curr)
                sdp = path1[:common_idx] + [curr] + path2[::-1]
                return sdp
            path2.append(curr)
            visited2.add(curr)
            curr = dep_heads[curr]
        
        # No path found, return empty
        return []
    
    def get_entity_dependents(self, dep_heads: List[int], entity_idx: int) -> List[int]:
        """Get direct dependents of an entity"""
        dependents = []
        for i, head in enumerate(dep_heads):
            if head == entity_idx:
                dependents.append(i)
        return dependents

class RelationDataset:
    def __init__(self, file_path: str, parser: DependencyParser, 
                 tokenizer, max_length=128, is_test=False):
        self.file_path = file_path
        self.parser = parser
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_test = is_test
        
        self.data = []
        self.labels = set()
        self.label2id = {}
        self.id2label = {}
        
        self._load_data()
        self._build_label_mapping()
    
    def _load_data(self):
        """Load data from JSONL file"""
        with open(self.file_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line.strip())
                self.data.append(item)
                if 'label' in item:
                    self.labels.add(item['label'])
    
    def _build_label_mapping(self):
        """Build label to ID mapping"""
        self.labels = sorted(list(self.labels))
        self.label2id = {label: i for i, label in enumerate(self.labels)}
        self.id2label = {i: label for label, i in self.label2id.items()}
    
    def _align_tokens_to_words(self, tokens, wordpieces):
        """Align PhoBERT wordpieces back to original tokens"""
        alignment = []
        wp_idx = 1  # Skip [CLS]
        
        for token in tokens:
            token_clean = token.replace('_', '')
            start_idx = wp_idx
            
            # Collect wordpieces for this token
            collected = ""
            while wp_idx < len(wordpieces) and not wordpieces[wp_idx].startswith('##'):
                if wordpieces[wp_idx] not in ['[CLS]', '[SEP]', '[PAD]']:
                    collected += wordpieces[wp_idx].replace('@@', '')
                    wp_idx += 1
                    if token_clean in collected:
                        break
                else:
                    wp_idx += 1
            
            # Handle subword tokens
            while wp_idx < len(wordpieces) and wordpieces[wp_idx].startswith('##'):
                collected += wordpieces[wp_idx].replace('##', '').replace('@@', '')
                wp_idx += 1
            
            alignment.append((start_idx, wp_idx))
        
        return alignment
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Extract entities and sentence
        sentence = item['sentence']
        
        # Handle different formats
        if 'entity_1' in item:
            e1_text = item['entity_1']['text']
            e2_text = item['entity_2']['text']
            e1_start = item['entity_1']['pos'][0]
            e2_start = item['entity_2']['pos'][0]
        else:
            e1_text = item['entities'][0]['text']
            e2_text = item['entities'][1]['text']
            e1_start = item['entities'][0]['start']
            e2_start = item['entities'][1]['start']
        
        # Parse with vnCoreNLP
        parsed = self.parser.parse(sentence)
        tokens = parsed['tokens']
        pos_tags = parsed['pos_tags']
        dep_heads = parsed['dep_heads']
        dep_rels = parsed['dep_rels']
        
        # Find entity positions in tokens
        e1_idx = self._find_entity_index(tokens, e1_text, e1_start, sentence)
        e2_idx = self._find_entity_index(tokens, e2_text, e2_start, sentence)
        
        # Find SDP
        sdp = self.parser.find_shortest_path(dep_heads, e1_idx, e2_idx)
        
        # Get entity dependents for enriched context
        e1_deps = self.parser.get_entity_dependents(dep_heads, e1_idx)
        e2_deps = self.parser.get_entity_dependents(dep_heads, e2_idx)
        
        # Build tree nodes (SDP + entity dependents)
        tree_nodes = set(sdp + e1_deps + e2_deps)
        
        # Tokenize with PhoBERT
        encoded = self.tokenizer(
            ' '.join(tokens),
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Get label
        label = -1 if self.is_test else self.label2id.get(item.get('label', 'Other'), 0)
        
        return {
            'input_ids': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0),
            'tokens': tokens,
            'pos_tags': pos_tags,
            'dep_heads': dep_heads,
            'dep_rels': dep_rels,
            'e1_idx': e1_idx,
            'e2_idx': e2_idx,
            'sdp': sdp,
            'tree_nodes': list(tree_nodes),
            'label': label,
            'sent_id': item.get('sent_id', idx)
        }
    
    def _find_entity_index(self, tokens, entity_text, char_start, sentence):
        """Find entity index in tokenized sequence"""
        entity_clean = entity_text.replace(' ', '_')
        
        for i, token in enumerate(tokens):
            if entity_clean == token or entity_text in token:
                return i
        
        # Fallback: find by position
        char_count = 0
        for i, token in enumerate(tokens):
            if char_count >= char_start:
                return i
            char_count += len(token.replace('_', ' ')) + 1
        
        return 0

def collate_fn(batch):
    """Custom collate function for DataLoader"""
    return {
        'input_ids': torch.stack([item['input_ids'] for item in batch]),
        'attention_mask': torch.stack([item['attention_mask'] for item in batch]),
        'labels': torch.tensor([item['label'] for item in batch]),
        'metadata': [
            {
                'tokens': item['tokens'],
                'pos_tags': item['pos_tags'],
                'dep_heads': item['dep_heads'],
                'dep_rels': item['dep_rels'],
                'e1_idx': item['e1_idx'],
                'e2_idx': item['e2_idx'],
                'sdp': item['sdp'],
                'tree_nodes': item['tree_nodes'],
                'sent_id': item['sent_id']
            }
            for item in batch
        ]
    }