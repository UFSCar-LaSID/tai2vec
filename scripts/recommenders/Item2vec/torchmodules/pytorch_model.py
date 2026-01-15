import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class Item2VecModel(nn.Module):
    def __init__(self, vocab_size, embedding_size, learning_rate, lr_decay, regularization=-1, loss_sum=True, normalize_after=False, big_innit=False):
        super(Item2VecModel, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.learning_rate = learning_rate
        self.lr_decay = lr_decay
        self.regularization = regularization
        self.loss_sum = loss_sum
        self.normalize_after = normalize_after
        self.big_innit = big_innit

        # Back to dense embeddings for standard Adam
        self.target_embedding = nn.Embedding(vocab_size, embedding_size)
        self.context_embedding = nn.Embedding(vocab_size, embedding_size)
        self._init_embeddings()
            
    def _init_embeddings(self):

        if self.big_innit:
            size = 1
        else:
            size = 1 / np.sqrt(self.embedding_size)

        nn.init.uniform_(self.target_embedding.weight, -size, size)
        nn.init.uniform_(self.context_embedding.weight, -size, size)

    def forward(self, target_items, context_items):
        target_emb = self.target_embedding(target_items)  
        context_emb = self.context_embedding(context_items)
        return torch.einsum("be,be->b", target_emb, context_emb) 
    
    def get_item_embeddings(self):
        """
        Returns the raw target and context embeddings.
        """
        t = self.target_embedding.weight.detach().cpu().numpy()
        c = self.context_embedding.weight.detach().cpu().numpy()
        return t, c
    
    def create_optimizer(self, max_epochs):
        # Use Adam; apply weight_decay only if regularization >= 0
        wd = 0 if self.regularization < 0 else self.regularization
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=wd)
        scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=self.lr_decay, total_iters=max_epochs)
        return optimizer, scheduler
    
    def compute_loss(self, targets, contexts, labels, weights=None):

        logits = self.forward(targets, contexts)
        bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction='none')
        if weights is None:
            return bce.mean()
        else:
            # Assume weights provided per pair; if size matches logits, it's per-element
            if weights.dim() == 1 and weights.numel() * 1 == logits.numel():
                # ambiguous; fallback to elementwise
                return (bce * weights).mean()
            else:
                return (bce.mean() * weights.mean())
    
    def get_loss(self, targets, contexts, weights=None):

        B, Kp1 = targets.shape
        # Flatten for embedding lookups
        logits = self.forward(targets.view(-1), contexts.view(-1)).view(B, Kp1)
        pos_logits = logits[:, 0]
        neg_logits = logits[:, 1:]
        # -log sigma(pos) = softplus(-pos)
        loss_pos = torch.nn.functional.softplus(-pos_logits)
        # -log sigma(-neg) = softplus(neg)
        loss_neg = torch.nn.functional.softplus(neg_logits)

        pair_loss = loss_pos + loss_neg.sum(dim=1)
            
        if weights is not None:
            pair_loss = pair_loss * weights

        return pair_loss.mean()