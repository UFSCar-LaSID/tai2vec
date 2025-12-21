import torch
import numpy as np
import time
import os
import scripts as kw
import pickle
import gc

class Item2VecTrainer:
    
    def __init__(self, parent, model):
        self.parent = parent
        self.model = model
        self.optimizer = None
        self.scheduler = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"PyTorch trainer initialized on device: {self.device}")
    
    def train(self, dataloader, data_repr):

        np.random.seed(kw.RANDOM_STATE)
        torch.manual_seed(kw.RANDOM_STATE)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(kw.RANDOM_STATE)
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        steps_per_epoch = len(dataloader) if hasattr(dataloader, '__len__') else 1
        total_iters = max(1, self.parent.epochs * steps_per_epoch)
        
        self.optimizer, self.scheduler = self.model.create_optimizer(max_epochs=total_iters)
        
        self.model.train()
        
        scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None
        use_amp = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 7
        
        for epoch in range(1, self.parent.epochs + 1):
            epoch_start_time = time.time()
            total_loss = 0.0
            num_batches = 0
            
            for batch_idx, (targets, contexts, labels, weights) in enumerate(dataloader):

                self.optimizer.zero_grad(set_to_none=True)

                targets = targets.to(self.device, non_blocking=True)
                contexts = contexts.to(self.device, non_blocking=True)
                weights = weights.to(self.device, non_blocking=True)
                
                if use_amp and scaler is not None:
                    with torch.amp.autocast('cuda'):
                        loss = self.model.get_loss(targets, contexts, weights=weights)
                    scaler.scale(loss).backward()
                    scaler.step(self.optimizer)
                    scaler.update()
                    if self.scheduler is not None:
                        self.scheduler.step()
                else:
                    loss = self.model.get_loss(targets, contexts, weights=weights)
                    loss.backward()
                    self.optimizer.step()
                    if self.scheduler is not None:
                        self.scheduler.step()
                
                if epoch == 0 and batch_idx == 0:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                    gc.collect()
                
                total_loss += loss.item()
                num_batches += 1
            
            epoch_time = time.time() - epoch_start_time
            avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
            time_per_step = epoch_time / num_batches if num_batches > 0 else 0.0
            
            print(f"Epoch {epoch}/{self.parent.epochs} - Time: {epoch_time:.1f}s, Steps: {num_batches}, Time/step: {time_per_step:.3f}s, Loss: {avg_loss:.4f}")

        torch.save(self.model.state_dict(), "best_model.pth")
        
        return self.model
