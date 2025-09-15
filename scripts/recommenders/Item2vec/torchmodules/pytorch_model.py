import torch
import torch.nn as nn
import torch.optim as optim

class Item2VecModel(nn.Module):
    def __init__(self, vocab_size, embedding_size, learning_rate, lr_decay, regularization=-1, init_strat='error'):
        super(Item2VecModel, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.learning_rate = learning_rate
        self.lr_decay = lr_decay
        self.regularization = regularization

        self.target_embedding = nn.Embedding(vocab_size, embedding_size, max_norm=1.0)
        self.context_embedding = nn.Embedding(vocab_size, embedding_size, max_norm=1.0)
        self._init_embeddings(init_strat)
            
    def _init_embeddings(self, init_strat):

        if init_strat == 'uniform_big':
            print("Using uniform_big initialization")
            nn.init.uniform_(self.target_embedding.weight, -0.5, 0.5)
            nn.init.uniform_(self.context_embedding.weight, -0.5, 0.5)
        elif init_strat == 'uniform_small':
            print("Using uniform_small initialization")
            size = 0.5 / self.embedding_size
            nn.init.uniform_(self.target_embedding.weight, -size, size)
            nn.init.uniform_(self.context_embedding.weight, -size, size)
        elif init_strat == 'xavier':
            print("Using xavier initialization")
            nn.init.xavier_uniform_(self.target_embedding.weight)
            nn.init.xavier_uniform_(self.context_embedding.weight)
        else:
            raise ValueError("Unknown initialization strategy")

    def forward(self, target_items, context_items):
        target_emb = self.target_embedding(target_items)  
        context_emb = self.context_embedding(context_items)
        return torch.einsum("be,be->b", target_emb, context_emb)  # dot product
    
    def get_item_embeddings(self):
        return ((self.target_embedding.weight + self.context_embedding.weight) / 2).detach().cpu().numpy()
        #return self.target_embedding.weight.detach().cpu().numpy()
    
    def create_optimizer(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.regularization if self.regularization > 0 else 0)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.lr_decay)
        return optimizer, scheduler
    
    def compute_loss(self, targets, contexts, labels, weights=None):
        logits = self.forward(targets, contexts)  # [B]
        bce_loss = nn.BCEWithLogitsLoss(reduction="none")(logits, labels)

        if weights is None:
            weights = torch.ones_like(labels)
        
        loss = (bce_loss * weights).mean()

        return loss
