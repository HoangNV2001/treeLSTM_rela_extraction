import torch
import torch.nn as nn
from transformers import AutoModel
from tree_lstm import TreeLSTM

class RelationExtractionModel(nn.Module):
    def __init__(self, num_labels, phobert_model='vinai/phobert-base-v2',
                 pos_vocab_size=50, dep_rel_vocab_size=40,
                 pos_embed_dim=32, dep_embed_dim=32, dist_embed_dim=32,
                 lstm_hidden=256, tree_lstm_hidden=256, 
                 dropout=0.5,  # INCREASED from 0.3
                 word_dropout=0.1,  # NEW: word-level dropout
                 use_layer_norm=True):  # NEW: layer normalization
        super(RelationExtractionModel, self).__init__()
        
        self.word_dropout = word_dropout
        self.use_layer_norm = use_layer_norm
        
        # PhoBERT encoder
        self.phobert = AutoModel.from_pretrained(phobert_model)
        self.phobert_dim = 768
        
        # Embedding layers with dropout
        self.pos_embedding = nn.Embedding(pos_vocab_size, pos_embed_dim)
        self.dep_rel_embedding = nn.Embedding(dep_rel_vocab_size, dep_embed_dim)
        self.embedding_dropout = nn.Dropout(dropout)
        
        # Distance embeddings
        self.dist_embedding = nn.Embedding(21, dist_embed_dim)
        
        # SDP position embedding
        self.sdp_embedding = nn.Embedding(3, 16)
        
        # Feature fusion with LayerNorm
        feature_dim = (self.phobert_dim + pos_embed_dim + dep_embed_dim + 
                      2 * dist_embed_dim + 16)
        
        self.feature_transform = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.LayerNorm(512) if use_layer_norm else nn.Identity(),
            nn.Tanh(),
            nn.Dropout(dropout)
        )
        
        # Sequence LSTM with increased dropout
        self.seq_lstm = nn.LSTM(512, lstm_hidden, batch_first=True, 
                                bidirectional=True, num_layers=1,
                                dropout=0)  # Use manual dropout instead
        self.lstm_dropout = nn.Dropout(dropout)
        
        # Tree-LSTM with smaller hidden size to reduce params
        tree_input_dim = 2 * lstm_hidden + dep_embed_dim
        self.tree_lstm = TreeLSTM(tree_input_dim, tree_lstm_hidden)
        
        # Classifier with more layers and dropout
        self.classifier = nn.Sequential(
            nn.Linear(tree_lstm_hidden, 512),
            nn.LayerNorm(512) if use_layer_norm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.LayerNorm(256) if use_layer_norm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_labels)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def bucket_distance(self, dist):
        """Bucket distance to range [-10, 10]"""
        return torch.clamp(dist + 10, 0, 20)
    
    def word_dropout_mask(self, tokens_len, device):
        """Apply word-level dropout during training"""
        if self.training and self.word_dropout > 0:
            mask = torch.bernoulli(
                torch.ones(tokens_len, device=device) * (1 - self.word_dropout)
            )
            return mask.unsqueeze(-1)
        return torch.ones(tokens_len, 1, device=device)
    
    def forward(self, input_ids, attention_mask, metadata):
        batch_size = input_ids.size(0)
        device = input_ids.device
        
        # PhoBERT encoding
        outputs = self.phobert(input_ids=input_ids, attention_mask=attention_mask)
        phobert_features = outputs.last_hidden_state
        
        batch_outputs = []
        
        for b in range(batch_size):
            meta = metadata[b]
            tokens = meta['tokens']
            pos_tags = meta['pos_tags']
            dep_rels = meta['dep_rels']
            dep_heads = meta['dep_heads']
            e1_idx = meta['e1_idx']
            e2_idx = meta['e2_idx']
            sdp = meta['sdp']
            tree_nodes = meta['tree_nodes']
            
            num_tokens = len(tokens)
            
            if len(tree_nodes) == 0:
                tree_nodes = [e1_idx, e2_idx]
            
            # Map tokens to PhoBERT
            token_features = phobert_features[b, 1:num_tokens+1, :]
            
            if token_features.size(0) < num_tokens:
                padding = torch.zeros(num_tokens - token_features.size(0), 
                                    self.phobert_dim, device=device)
                token_features = torch.cat([token_features, padding], dim=0)
            else:
                token_features = token_features[:num_tokens]
            
            # Apply word dropout
            word_mask = self.word_dropout_mask(num_tokens, device)
            token_features = token_features * word_mask
            
            # Create embeddings
            pos_ids = torch.tensor([hash(pos) % 50 for pos in pos_tags], device=device)
            dep_ids = torch.tensor([hash(rel) % 40 for rel in dep_rels], device=device)
            
            pos_emb = self.embedding_dropout(self.pos_embedding(pos_ids))
            dep_emb = self.embedding_dropout(self.dep_rel_embedding(dep_ids))
            
            # Distance embeddings
            dist_e1 = torch.tensor([i - e1_idx for i in range(num_tokens)], device=device)
            dist_e2 = torch.tensor([i - e2_idx for i in range(num_tokens)], device=device)
            
            dist_e1_emb = self.dist_embedding(self.bucket_distance(dist_e1))
            dist_e2_emb = self.dist_embedding(self.bucket_distance(dist_e2))
            
            # SDP position
            sdp_pos = torch.zeros(num_tokens, dtype=torch.long, device=device)
            if len(sdp) > 0:
                sdp_mid = len(sdp) // 2
                for i, idx in enumerate(sdp):
                    if idx < num_tokens:
                        if i < sdp_mid:
                            sdp_pos[idx] = 1
                        else:
                            sdp_pos[idx] = 2
            sdp_emb = self.sdp_embedding(sdp_pos)
            
            # Concatenate features
            combined = torch.cat([
                token_features[:num_tokens],
                pos_emb,
                dep_emb,
                dist_e1_emb,
                dist_e2_emb,
                sdp_emb
            ], dim=-1)
            
            # Transform
            transformed = self.feature_transform(combined)
            
            # Sequence LSTM
            transformed = transformed.unsqueeze(0)
            seq_output, _ = self.seq_lstm(transformed)
            seq_output = self.lstm_dropout(seq_output.squeeze(0))
            
            # Prepare Tree-LSTM input
            tree_features = []
            valid_tree_nodes = []
            for node_idx in tree_nodes:
                if node_idx < num_tokens:
                    node_dep_emb = dep_emb[node_idx]
                    node_seq_features = seq_output[node_idx]
                    tree_input = torch.cat([node_seq_features, node_dep_emb], dim=-1)
                    tree_features.append(tree_input)
                    valid_tree_nodes.append(node_idx)
            
            if len(tree_features) == 0:
                tree_features = [
                    torch.cat([seq_output[e1_idx], dep_emb[e1_idx]], dim=-1),
                    torch.cat([seq_output[e2_idx], dep_emb[e2_idx]], dim=-1)
                ]
                valid_tree_nodes = [e1_idx, e2_idx]
            
            tree_features = torch.stack(tree_features)
            
            # Find root
            root_idx = None
            if len(sdp) > 0:
                sdp_middle = sdp[len(sdp) // 2]
                if sdp_middle in valid_tree_nodes:
                    root_idx = sdp_middle
            
            if root_idx is None:
                root_idx = e1_idx if e1_idx in valid_tree_nodes else valid_tree_nodes[0]
            
            # Tree-LSTM
            root_hidden, _ = self.tree_lstm(tree_features, dep_heads, valid_tree_nodes, root_idx)
            
            batch_outputs.append(root_hidden)
        
        # Stack batch
        batch_hidden = torch.stack(batch_outputs)
        
        # Classifier
        logits = self.classifier(batch_hidden)
        
        return logits