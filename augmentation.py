import json
import random
from typing import List, Dict
import re

class RelationDataAugmenter:
    """Data augmentation for Vietnamese relation extraction"""
    
    def __init__(self, swap_prob=0.3, delete_prob=0.1):
        self.swap_prob = swap_prob
        self.delete_prob = delete_prob
        
        # Vietnamese stopwords that can be safely removed
        self.stopwords = {'và', 'hoặc', 'nhưng', 'mà', 'thì', 'nên', 
                         'của', 'cho', 'với', 'từ', 'tại', 'trên', 'trong'}
    
    def entity_swap(self, item: Dict) -> Dict:
        """Swap entity positions and reverse relation direction"""
        new_item = item.copy()
        
        # Check if relation is directional
        if 'entity_1' in item:
            e1 = item['entity_1'].copy()
            e2 = item['entity_2'].copy()
            
            new_item['entity_1'] = e2
            new_item['entity_2'] = e1
            
            # Reverse label if directional
            if 'label' in item:
                label = item['label']
                if '(e1,e2)' in label:
                    new_item['label'] = label.replace('(e1,e2)', '(e2,e1)')
                elif '(e2,e1)' in label:
                    new_item['label'] = label.replace('(e2,e1)', '(e1,e2)')
        
        return new_item
    
    def token_shuffle_near_entities(self, item: Dict) -> Dict:
        """Shuffle tokens near entities (excluding entities themselves)"""
        new_item = item.copy()
        sentence = item['sentence']
        tokens = sentence.split()
        
        if 'entity_1' in item:
            e1_text = item['entity_1']['text']
            e2_text = item['entity_2']['text']
        else:
            e1_text = item['entities'][0]['text']
            e2_text = item['entities'][1]['text']
        
        # Find entity positions
        e1_pos = sentence.find(e1_text)
        e2_pos = sentence.find(e2_text)
        
        # Don't shuffle if entities are too close
        if abs(e1_pos - e2_pos) < len(e1_text) + len(e2_text) + 3:
            return new_item
        
        # Shuffle tokens between entities (if applicable)
        if random.random() < self.swap_prob:
            start = min(e1_pos, e2_pos) + max(len(e1_text), len(e2_text))
            end = max(e1_pos, e2_pos)
            
            if end > start:
                middle_part = sentence[start:end].strip()
                middle_tokens = middle_part.split()
                
                if len(middle_tokens) > 2:
                    random.shuffle(middle_tokens)
                    new_middle = ' '.join(middle_tokens)
                    new_item['sentence'] = sentence[:start] + ' ' + new_middle + ' ' + sentence[end:]
        
        return new_item
    
    def context_word_dropout(self, item: Dict) -> Dict:
        """Randomly drop context words (not entities)"""
        new_item = item.copy()
        sentence = item['sentence']
        tokens = sentence.split()
        
        if 'entity_1' in item:
            e1_text = item['entity_1']['text']
            e2_text = item['entity_2']['text']
        else:
            e1_text = item['entities'][0]['text']
            e2_text = item['entities'][1]['text']
        
        new_tokens = []
        i = 0
        while i < len(tokens):
            token = tokens[i]
            
            # Check if token is part of entity
            is_entity_token = (token in e1_text or token in e2_text or
                             e1_text.startswith(token) or e2_text.startswith(token))
            
            # Drop context words with probability
            if not is_entity_token and token.lower() in self.stopwords:
                if random.random() > self.delete_prob:
                    new_tokens.append(token)
            else:
                new_tokens.append(token)
            
            i += 1
        
        new_item['sentence'] = ' '.join(new_tokens)
        return new_item
    
    def synonym_replacement_context(self, item: Dict) -> Dict:
        """Replace context words with synonyms (Vietnamese specific)"""
        # Simplified synonym dictionary for common Vietnamese words
        synonyms = {
            'lớn': ['to', 'rộng', 'khổng lồ'],
            'nhỏ': ['bé', 'nhỏ bé', 'tí hon'],
            'tốt': ['hay', 'giỏi', 'xuất sắc'],
            'xấu': ['tệ', 'kém'],
            'nhiều': ['đông đảo', 'phong phú'],
            'ít': ['thiếu', 'khan hiếm'],
            'nhanh': ['mau', 'nhanh chóng'],
            'chậm': ['chậm chạp', 'lâu'],
        }
        
        new_item = item.copy()
        sentence = item['sentence']
        
        for word, syns in synonyms.items():
            if word in sentence and random.random() < 0.2:
                new_item['sentence'] = sentence.replace(word, random.choice(syns), 1)
                break
        
        return new_item
    
    def augment_dataset(self, input_file: str, output_file: str, 
                       augment_ratio: float = 0.3, methods: List[str] = None):
        """
        Augment dataset and save to file
        
        Args:
            input_file: Path to input JSONL file
            output_file: Path to output JSONL file
            augment_ratio: Ratio of augmented samples to add (0.3 = 30% more data)
            methods: List of augmentation methods to use
        """
        if methods is None:
            methods = ['entity_swap', 'context_dropout']
        
        # Load original data
        original_data = []
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                original_data.append(json.loads(line.strip()))
        
        print(f"Original dataset size: {len(original_data)}")
        
        # Generate augmented samples
        augmented_data = []
        num_to_augment = int(len(original_data) * augment_ratio)
        
        for _ in range(num_to_augment):
            item = random.choice(original_data)
            method = random.choice(methods)
            
            if method == 'entity_swap':
                aug_item = self.entity_swap(item)
            elif method == 'context_dropout':
                aug_item = self.context_word_dropout(item)
            elif method == 'token_shuffle':
                aug_item = self.token_shuffle_near_entities(item)
            elif method == 'synonym':
                aug_item = self.synonym_replacement_context(item)
            else:
                aug_item = item
            
            augmented_data.append(aug_item)
        
        # Combine original and augmented
        combined_data = original_data + augmented_data
        random.shuffle(combined_data)
        
        # Save to file
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in combined_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        print(f"Augmented dataset size: {len(combined_data)}")
        print(f"Added {len(augmented_data)} augmented samples")
        print(f"Saved to {output_file}")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='Input JSONL file')
    parser.add_argument('--output', type=str, required=True, help='Output JSONL file')
    parser.add_argument('--ratio', type=float, default=0.3, help='Augmentation ratio')
    parser.add_argument('--methods', nargs='+', 
                       default=['entity_swap', 'context_dropout'],
                       help='Augmentation methods to use')
    args = parser.parse_args()
    
    augmenter = RelationDataAugmenter()
    augmenter.augment_dataset(args.input, args.output, args.ratio, args.methods)