import torch
import torch.nn as nn
import torch.optim as optim

class Item2VecModel(nn.Module):
    def __init__(self, vocab_size, embedding_size, learning_rate, lr_decay, regularization=-1):
        super(Item2VecModel, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.learning_rate = learning_rate
        self.lr_decay = lr_decay
        self.regularization = regularization

        # Back to dense embeddings for standard Adam
        self.target_embedding = nn.Embedding(vocab_size, embedding_size)
        self.context_embedding = nn.Embedding(vocab_size, embedding_size)
        self._init_embeddings()
            
    def _init_embeddings(self):

        size = 0.1
        nn.init.uniform_(self.target_embedding.weight, -size, size)
        nn.init.uniform_(self.context_embedding.weight, -size, size)

    def forward(self, target_items, context_items):
        target_emb = self.target_embedding(target_items)  
        context_emb = self.context_embedding(context_items)
        return torch.einsum("be,be->b", target_emb, context_emb) 
    
    def get_item_embeddings(self):
        return ((self.target_embedding.weight + self.context_embedding.weight) / 2).detach().cpu().numpy()
    
    def create_optimizer(self, max_epochs):
        # Use Adam; apply weight_decay only if regularization >= 0
        wd = 0 if self.regularization < 0 else self.regularization
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=wd)
        scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=self.lr_decay, total_iters=max_epochs)
        return optimizer, scheduler
    
    def compute_loss(self, targets, contexts, labels, weights=None):

        logits = self.forward(targets, contexts)

        bce_loss = nn.BCEWithLogitsLoss(reduction="mean", weight=weights)(logits, labels)

        return bce_loss