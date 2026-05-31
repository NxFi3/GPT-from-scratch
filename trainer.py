import os 
import torch 
from torch.cuda.amp import autocast
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
import torch.nn.functional as F
import time

def is_bfloat16_supported():
    if torch.cuda.is_available():
        return torch.cuda.is_bf16_supported()
    return False

class EWC:
    def __init__(self, model, dataloader, device, importance=1000):
        self.model = model
        self.device = device
        self.importance = importance
        
     
        self.params = {
            n: p.clone().detach() 
            for n, p in model.named_parameters() 
            if p.requires_grad
        }
        
        # compute Fisher Information Matrix
        self.fisher = {}
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.fisher[n] = torch.zeros_like(p)
        
        model.train()
        n_batches = min(len(dataloader), 50)  
        
        for i, batch in enumerate(dataloader):
            if i >= n_batches:
                break
            x, y = batch
            x, y = x.to(device), y.to(device)
            
            model.zero_grad()
            logits, _ = model(x, padding_mask=None)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                y.view(-1),
                ignore_index=-100
            )
            loss.backward()
            
            for n, p in model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    self.fisher[n] += p.grad.pow(2) / n_batches
    
    def penalty(self, model):
        loss = 0
        for n, p in model.named_parameters():
            if n in self.fisher:
                loss += (self.fisher[n] * (p - self.params[n]).pow(2)).sum()
        return self.importance * loss

class Trainer:
    def __init__(self, model, dataloader, lr=2e-4, weight_decay=0.01, 
                 accum_steps=2, checkpoint_step=1000, checkpoint_dir='checkpoints',device='cuda'): # change device to cpu if you dont have NVIDIA GPU
        self.model = model
        self.accum_steps = accum_steps
        self.save_every = checkpoint_step
        self.global_step = 0  
        self.device = torch.device(device)
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.checkpoint_dir = checkpoint_dir
        self.dataloader = dataloader
        total_steps = len(dataloader)
        
        warmup_steps = max(200, int(0.03 * total_steps))
        
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=lr, 
            weight_decay=weight_decay
        )
        
        self.warmup = LinearLR(
            self.optimizer, 
            start_factor=0.2, 
            end_factor=1.0, 
            total_iters=warmup_steps
        )
        
        self.cosine = CosineAnnealingLR(
            self.optimizer, 
            T_max=max(1, total_steps - warmup_steps)
        )
        
        self.scheduler = SequentialLR(
            self.optimizer, 
            schedulers=[self.warmup, self.cosine], 
            milestones=[warmup_steps]
        )
    
    def load_checkpoint(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.scheduler.load_state_dict(ckpt["scheduler"])
        self.global_step = ckpt["step"]  
        print(f"✅ Resumed from step {self.global_step}")
    
    def Train(self, num_epochs=1, model_checkpoint=None, 
              use_ewc=False, ewc_checkpoint=None, old_phase_dataloader=None):
        
        ewc = None
        
        if use_ewc:
            ewc = EWC(self.model, old_phase_dataloader, self.device, importance=1500)
            if ewc_checkpoint is not None:
                ewc = torch.load(ewc_checkpoint)
            if model_checkpoint is not None:
                self.load_checkpoint(model_checkpoint)   
        
        step_losses = []
        step_numbers = []
        lr_history = []
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            self.optimizer.zero_grad(set_to_none=True)
            epoch_loss = 0 
            
            for step, batch in enumerate(self.dataloader):


                x, y = batch
                x = x.to(self.device)
                y = y.to(self.device)

                # packed dataset → no padding needed
                padding = None

                # =========================
                # forward
                # =========================
                with autocast(dtype=torch.bfloat16 if is_bfloat16_supported() else torch.float16):
                    logits, _ =self.model(x, padding_mask=padding)

                    base_loss = F.cross_entropy(
                        logits.view(-1, logits.size(-1)),
                        y.view(-1),
                        ignore_index=-100
                    )
                    
                    if ewc is not None:
                        ewc_loss = ewc.penalty(self.model)
                        total_loss = base_loss + ewc_loss 
                    else:
                        total_loss = base_loss 
                    
                    loss = total_loss / self.accum_steps
                
                loss.backward()
     
                if (step + 1) % self.accum_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.global_step += 1
                    
                    lr_now = self.optimizer.param_groups[0]['lr']
                    
                    step_losses.append(base_loss.item())
                    step_numbers.append(self.global_step)
                    lr_history.append(lr_now)
                    
                    if self.global_step % 10 == 0:
                        if ewc is not None:
                            print(f"[{self.global_step}] base_loss={base_loss.item():.4f} "
                                  f"ewc_loss={ewc_loss.item():.4f} total={total_loss.item():.4f} "
                                  f"lr={lr_now:.2e}")
                        else:
                            print(f"[{self.global_step}] loss={base_loss.item():.4f} "
                                  f"lr={lr_now:.2e}")
                    
                    if self.global_step % self.save_every == 0:
                        torch.save({
                            "model": self.model.state_dict(),
                            "optimizer": self.optimizer.state_dict(),
                            "scheduler": self.scheduler.state_dict(),
                            "step": self.global_step,
                            "epoch": epoch
                        }, f"{self.checkpoint_dir}/ckpt_{self.global_step}.pt")
                        
                        if ewc is not None:
                            torch.save(ewc, f"{self.checkpoint_dir}/ewc_{self.global_step}.pt")
                        
                        print(f"💾 Saved checkpoint at step {self.global_step}")
                
                epoch_loss += base_loss.item()
            

            avg_epoch_loss = epoch_loss / len(self.dataloader)
            print(f"\nEpoch {epoch} done | avg_loss={avg_epoch_loss:.4f}")
        

        if ewc is not None:
            torch.save(ewc, f"{self.checkpoint_dir}/ewc_final.pt")
            print("✅ EWC saved!")
        
        torch.save({
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "step": self.global_step
        }, f"{self.checkpoint_dir}/final.pt")
        
        torch.save({
            "step_losses": step_losses,
            "step_numbers": step_numbers,
            "lr_history": lr_history,
            "total_steps": self.global_step,
            "total_time": time.time() - start_time
        }, f"{self.checkpoint_dir}/train_logs.pt")
        
        print("✅ TRAINING COMPLETE")