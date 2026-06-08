import torch
from torch import optim
from torchmetrics import F1Score, Accuracy
import numpy as np
from tqdm import tqdm
import os

from Models import BaseClassifierDEAP
from data import build_loaders
from configs import *

import torch._dynamo
torch._dynamo.config.suppress_errors = True
os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "OFF"
os.environ["TORCH_LOGS"] = "-dynamo,-inductor"
os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"
os.environ["TRITON_PRINT_AUTOTUNE"] = "0"

os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "OFF"
os.environ["TORCH_LOGS"] = "-dynamo,-inductor"
os.environ["TRITON_PRINT_AUTOTUNE"] = "0"

# 2. Block python warnings globally
import warnings
import logging
import sys
warnings.filterwarnings("ignore")
logging.getLogger("torch").setLevel(logging.ERROR)

# 3. Hijack and mute Python's standard logging streams
logging.getLogger("torch").setLevel(logging.ERROR)

class NoiseFilter(logging.Filter):
    def filter(self, record):
        # Drop anything containing compiler or framework keywords
        msg = record.getMessage().lower()
        keywords = ["dynamo", "mamba", "speculate", "subgraph", "trampoline", "autocast"]
        return not any(k in msg for k in keywords)

# Apply the text filter to standard logging outputs
root_logger = logging.getLogger()
root_logger.addFilter(NoiseFilter())

# 4. Hijack the low-level stderr stream where C++ compiler prints directly
class StreamSilencer:
    def __init__(self, original_stream):
        self.original_stream = original_stream
    def write(self, data):
        # Ignore raw text printed by C++ libraries
        data_lower = data.lower()
        if any(k in data_lower for k in ["_posix_c_source", "redefined", "warning", "error"]):
            return
        self.original_stream.write(data)
    def flush(self):
        self.original_stream.flush()

sys.stderr = StreamSilencer(sys.stderr)

def get_scheduler(optimizer, extra=None):
    
    if scheduler_type == 'onecycle':
        return optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer,
            max_lr=max_lr,
            div_factor=int(max_lr / lr),
            final_div_factor=10,
            epochs=epochs,
            steps_per_epoch=extra,
            anneal_strategy='cos'
        )
        
    elif scheduler_type == 'cos':
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer,
            T_max=epochs,
            eta_min=lr/10
        )
        
    elif scheduler_type == 'step':
        return optim.lr_scheduler.StepLR(
            optimizer=optimizer,
            gamma=gamma,
            step_size=step_size
        )

def train():
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    supported = torch.cuda.is_bf16_supported()
    cast_type = torch.bfloat16 if supported else torch.float16
    torch.backends.cudnn.benchmark = True
    
    trndta, trnwt, valdta, valwt = build_loaders('DEAP')
    print(device, supported)
    print(trnwt)
    print(valwt)
    best_f1 = float('-inf')
    os.makedirs('saved', exist_ok=True)
    
    model = BaseClassifierDEAP().to(device)
    optimizer = optim.AdamW(params=model.parameters(),lr=lr, weight_decay=wdc)
    scheduler = get_scheduler(optimizer=optimizer,extra=len(trndta))
    scaler = torch.cuda.amp.GradScaler()
    loss_fun_emot = torch.nn.CrossEntropyLoss(weight=torch.tensor(trnwt),label_smoothing=0.1).float().to(device)
    loss_fun_pats = torch.nn.CrossEntropyLoss().float().to(device)
    
    for epoch in range(epochs):
        pbar = tqdm(trndta, desc=f"Epoch [{epoch}/{epochs}] Train")
        for data in pbar:
            
            model.train()
            eeg = data['eeg'].to(device, non_blocking=True)
            gsr = data['gsr'].to(device, non_blocking=True)
            bvp = data['bvp'].to(device, non_blocking=True)
            emotion_label = data['emotion_label'].to(device, non_blocking=True)
            patient_label = data['patient_label'].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            
            with torch.autocast(device_type=device,dtype=cast_type):
                preds = model((eeg,gsr,bvp),epoch)
                
                if not do_uda:
                    loss = loss_fun_emot(preds.float(), emotion_label.long())
                else: 
                    emo_logits, subj_logits, z_emo, z_subj = preds 

                    loss_emo = loss_fun_emot(emo_logits.float(), emotion_label.long())
                    loss_subj = loss_fun_pats(subj_logits.float(), patient_label.long())

                    z_emo_norm = torch.nn.functional.normalize(z_emo, p=2, dim=1)
                    z_subj_norm = torch.nn.functional.normalize(z_subj, p=2, dim=1)
                    loss_ortho = torch.mean((z_emo_norm * z_subj_norm).sum(dim=1) ** 2)
                    loss = loss_emo + loss_subj + (ortho_wt * loss_ortho)
                
            if supported:
                loss.backward()
                optimizer.step()
            else:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
            pbar.set_postfix(loss=f"{loss.item():.4f}")
            if scheduler_type == 'onecycle': scheduler.step()
        if scheduler_type in ['step','cos']: scheduler.step()
        
        model.eval()
        acc_metric = Accuracy(task="multiclass", num_classes=4).to(device)
        f1_metric = F1Score(task="multiclass", num_classes=4, average="macro").to(device)
        loss_fun_val = torch.nn.CrossEntropyLoss(weight=torch.tensor(np.max(valwt)/valwt)).float().to(device)
        
        with torch.no_grad():
            val_pbar = tqdm(valdta, desc=f"Epoch [{epoch}/{epochs}] Val", leave=False)
            for data in val_pbar:
                eeg = data['eeg'].to(device, non_blocking=True)
                gsr = data['gsr'].to(device, non_blocking=True)
                bvp = data['bvp'].to(device, non_blocking=True)
                emotion_label = data['emotion_label'].to(device, non_blocking=True)
                
                with torch.autocast(device_type=device, dtype=cast_type):
                    preds = model((eeg, gsr, bvp), 0.0) 
                    emo_logits = preds[0] if do_uda else preds
                    val_loss = loss_fun_val(emo_logits, emotion_label).item()
                    
                val_pbar.set_postfix(v_loss=f"{val_loss:.4f}")
                acc_metric.update(emo_logits, emotion_label)
                f1_metric.update(emo_logits, emotion_label)
                
        print(f"Epoch {epoch} | Acc: {acc_metric.compute():.4f} | F1: {f1_metric.compute():.4f}")
        val_f1 = f1_metric.compute()

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'f1': best_f1,
            }, os.path.join('saved',"best_emotion_model.pth"))
            print(f"*** New Best F1: {best_f1:.4f} - Model Saved ***")
            
        acc_metric.reset()
        f1_metric.reset()
        
if __name__ == '__main__': train()