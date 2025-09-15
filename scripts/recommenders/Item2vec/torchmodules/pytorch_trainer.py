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
        
        self.optimizer, self.scheduler = self.model.create_optimizer()
        
        self.model.train()
        
        scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None
        use_amp = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 7
        
        for epoch in range(1, self.parent.epochs+1):
            epoch_start_time = time.time()
            total_loss = 0.0
            num_batches = 0
            
            for batch_idx, (targets, contexts, labels, weights) in enumerate(dataloader):

                targets = targets.flatten().to(self.device, non_blocking=True)
                contexts = contexts.flatten().to(self.device, non_blocking=True)
                labels = labels.flatten().to(self.device, non_blocking=True)
                weights = weights.flatten().to(self.device, non_blocking=True)
                
                if use_amp and scaler is not None:
                    with torch.amp.autocast('cuda'):
                        loss = self.model.compute_loss(targets, contexts, labels, weights)
                    
                    scaler.scale(loss).backward()
                    scaler.step(self.optimizer)
                    scaler.update()
                else:
                    loss = self.model.compute_loss(targets, contexts, labels, weights)
                    loss.backward()
                    self.optimizer.step()
                
                if epoch == 0 and batch_idx == 0:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                    gc.collect()
                
                total_loss += loss.item()
                num_batches += 1
            
            # Update learning rate
            self.scheduler.step()
            
            # Print clean epoch summary
            epoch_time = time.time() - epoch_start_time
            avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
            time_per_step = epoch_time / num_batches if num_batches > 0 else 0.0
            
            print(f"Epoch {epoch+1}/{self.parent.epochs} - Time: {epoch_time:.1f}s, Steps: {num_batches}, Time/step: {time_per_step:.3f}s, Loss: {avg_loss:.4f}")
            
            # Save embeddings callback
            if (epoch == 5) or (epoch == 10) or (epoch == 20) or (epoch == 50) or (epoch == 100) or (epoch == 150) or (epoch == 200):
                with torch.no_grad():
                    self._save_embeddings(epoch, data_repr)
        
        # Cleanup
        del dataloader
        if use_amp and scaler is not None:
            del scaler
        
        return self.model
    
    def _save_embeddings(self, epoch, data_repr):

        path_components = os.path.normpath(self.parent.embedding_dir).split(os.sep)

        if len(path_components) > 2 and path_components[2] == 'validation':
            embedding_dir = self.parent.embedding_dir + "@epochs={}".format(epoch)
        else:
            embedding_dir = self.parent.embedding_dir

        os.makedirs(embedding_dir, exist_ok=True)
        item_embeddings = self.model.get_item_embeddings()
        np.save(os.path.join(embedding_dir, kw.FILE_ITEMS_EMBEDDINGS), item_embeddings)
        pickle.dump(data_repr, open(os.path.join(embedding_dir, kw.FILE_SPARSE_REPR), 'wb'))
