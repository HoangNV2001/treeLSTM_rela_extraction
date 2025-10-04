import torch
import torch.nn as nn
from typing import List, Dict

class TreeLSTMCell(nn.Module):
    """Child-Sum Tree-LSTM Cell"""
    
    def __init__(self, input_dim, hidden_dim):
        super(TreeLSTMCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Input gate
        self.W_i = nn.Linear(input_dim, hidden_dim)
        self.U_i = nn.Linear(hidden_dim, hidden_dim, bias=False)
        
        # Forget gates (one per child)
        self.W_f = nn.Linear(input_dim, hidden_dim)
        self.U_f = nn.Linear(hidden_dim, hidden_dim, bias=False)
        
        # Output gate
        self.W_o = nn.Linear(input_dim, hidden_dim)
        self.U_o = nn.Linear(hidden_dim, hidden_dim, bias=False)
        
        # Cell input
        self.W_u = nn.Linear(input_dim, hidden_dim)
        self.U_u = nn.Linear(hidden_dim, hidden_dim, bias=False)
    
    def forward(self, x, child_h, child_c):
        """
        Args:
            x: input vector (input_dim,)
            child_h: list of children hidden states [(hidden_dim,), ...]
            child_c: list of children cell states [(hidden_dim,), ...]
        Returns:
            h: hidden state (hidden_dim,)
            c: cell state (hidden_dim,)
        """
        if len(child_h) == 0:
            # Leaf node
            child_h_sum = torch.zeros(self.hidden_dim, device=x.device)
        else:
            child_h_sum = torch.stack(child_h).sum(dim=0)
        
        # Input gate
        i = torch.sigmoid(self.W_i(x) + self.U_i(child_h_sum))
        
        # Output gate
        o = torch.sigmoid(self.W_o(x) + self.U_o(child_h_sum))
        
        # Cell input
        u = torch.tanh(self.W_u(x) + self.U_u(child_h_sum))
        
        # Forget gates and cell state
        if len(child_c) == 0:
            c = i * u
        else:
            f_gates = []
            for child_h_k in child_h:
                f_k = torch.sigmoid(self.W_f(x) + self.U_f(child_h_k))
                f_gates.append(f_k)
            
            forget_sum = sum(f_k * c_k for f_k, c_k in zip(f_gates, child_c))
            c = i * u + forget_sum
        
        # Hidden state
        h = o * torch.tanh(c)
        
        return h, c


class TreeLSTM(nn.Module):
    """Tree-LSTM for Dependency Tree"""
    
    def __init__(self, input_dim, hidden_dim):
        super(TreeLSTM, self).__init__()
        self.cell = TreeLSTMCell(input_dim, hidden_dim)
        self.hidden_dim = hidden_dim
    
    def forward(self, node_features, dep_heads, tree_nodes, root_idx=None):
        """
        Args:
            node_features: tensor of shape (num_nodes, input_dim)
            dep_heads: list of parent indices for each node
            tree_nodes: list of node indices in the tree
            root_idx: index of root node (if None, find automatically)
        Returns:
            root_hidden: hidden state of root node
            all_hidden: dict mapping node_idx -> hidden state
        """
        num_nodes = len(tree_nodes)
        node_map = {idx: i for i, idx in enumerate(tree_nodes)}
        
        # Build children lists
        children = {idx: [] for idx in tree_nodes}
        for node_idx in tree_nodes:
            parent_idx = dep_heads[node_idx]
            if parent_idx in node_map and parent_idx != node_idx:
                children[parent_idx].append(node_idx)
        
        # Find root if not provided
        if root_idx is None:
            # Root is node with no parent in tree or parent outside tree
            for node_idx in tree_nodes:
                parent_idx = dep_heads[node_idx]
                if parent_idx not in node_map or parent_idx == -1:
                    root_idx = node_idx
                    break
            if root_idx is None:
                root_idx = tree_nodes[len(tree_nodes) // 2]
        
        # Bottom-up traversal
        hidden_states = {}
        cell_states = {}
        
        def compute_node(node_idx):
            if node_idx in hidden_states:
                return hidden_states[node_idx], cell_states[node_idx]
            
            # Recursively compute children first
            child_h = []
            child_c = []
            for child_idx in children[node_idx]:
                c_h, c_c = compute_node(child_idx)
                child_h.append(c_h)
                child_c.append(c_c)
            
            # Compute this node
            x = node_features[node_map[node_idx]]
            h, c = self.cell(x, child_h, child_c)
            
            hidden_states[node_idx] = h
            cell_states[node_idx] = c
            
            return h, c
        
        # Compute from root
        root_h, root_c = compute_node(root_idx)
        
        return root_h, hidden_states