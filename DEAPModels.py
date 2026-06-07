import torch
from torch import nn
import torch.nn.functional as f
from mamba_ssm import Mamba

from configs import *

class getActivation(nn.Module):
    
    def __init__(self, act:str | None='relu'):
        super().__init__()
        self.activationFunction = lambda x : x
        act = act.lower()
        
        if act=='gelu': self.activationFunction = nn.GELU(approximate='tanh')
        elif act=='lrelu': self.activationFunction = nn.LeakyReLU(negative_slope=0.2)
        elif act=='elu': self.activationFunction = nn.ELU()
        elif act=='prelu': self.activationFunction = nn.PReLU()
        elif act=='sigmoid': self.activationFunction = nn.Sigmoid()
        elif act=='tanh': self.activationFunction = nn.Tanh()
        elif act=='relu': self.activationFunction = nn.ReLU()
        elif act=='silu': self.activationFunction = nn.SiLU()
        elif act=='softplus': self.activationFunction = nn.Softplus()
        
    def forward(self, x)->torch.Tensor:
        return self.activationFunction(x)
    
class SqueezeExciteAttention(nn.Module):
    
    def __init__(self, dim, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bottle_neck = nn.Sequential(
            nn.Linear(dim, dim//4),
            getActivation(actvsea),
            nn.Linear(dim//4, dim),
            nn.Sigmoid()
        )
        
    def forward(self, x:torch.Tensor): return x * (self.bottle_neck(x.mean(-1))).unsqueeze(-1)
        
class EEGBandExtractor(nn.Module):
    def __init__(self, k, d, k2):
        super().__init__()
        self.extractor = nn.Sequential(
            nn.ReplicationPad1d((k-1)*d // 2),
            nn.Conv1d(in_channels=32,out_channels=32, kernel_size=k, dilation=d, groups=32),
            getActivation(actveeg),
            nn.ReplicationPad1d(k2 // 2),
            nn.Conv1d(in_channels=32,out_channels=32, kernel_size=k2, groups=32),
            getActivation(actveeg),
            SqueezeExciteAttention(dim=32)
        )
        
    def forward(self, x): return self.extractor(x)

class EEGEncoder(nn.Module):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.alpha_band_extractor = EEGBandExtractor(k=5,d=6,k2=7)
        self.theta_band_extractor = EEGBandExtractor(k=5,d=3,k2=3)
        self.beta_band_extractor = EEGBandExtractor(k=5,d=2,k2=3)
        self.gamma_band_extractor = EEGBandExtractor(k=3,d=1,k2=3)
        
        self.mixer = nn.Sequential(
            nn.Conv1d(in_channels=32*4,out_channels=32*3,kernel_size=1),
            getActivation(actveeg),
            nn.Conv1d(in_channels=32*3,out_channels=32*2,kernel_size=1),
            getActivation(actveeg),
            nn.Conv1d(in_channels=32*2,out_channels=mdim,kernel_size=1),
        )
        
    def forward(self,x):
        
        alpha = self.alpha_band_extractor(x)
        beta = self.beta_band_extractor(x)
        gamma = self.gamma_band_extractor(x)
        theta = self.theta_band_extractor(x)
        feats = torch.cat([alpha,beta,gamma,theta],dim=-2)
        return self.mixer(feats).mT.contiguous()

class AuxilliaryEncoder(nn.Module):
    def __init__(self, main_dils, sec_k):
        super().__init__()
        
        self.main_path = nn.Sequential(
            nn.ReplicationPad1d(main_dils[0]*2),
            nn.Conv1d(in_channels=1,out_channels=mdimp//4,dilation=main_dils[0],kernel_size=5),
            getActivation(actaux),
            
            nn.ReplicationPad1d(main_dils[1]*2),
            nn.Conv1d(in_channels=mdimp//4,out_channels=mdimp//2,dilation=main_dils[1],kernel_size=5),
            getActivation(actaux),
            
            nn.ReplicationPad1d(main_dils[2]*2),
            nn.Conv1d(in_channels=mdimp//2,out_channels=mdimp,dilation=main_dils[2],kernel_size=5),
            getActivation(actaux)
        )
        self.sec_path = nn.Sequential(
            nn.ReplicationPad1d(sec_k[0]//2),
            nn.Conv1d(in_channels=1,out_channels=4,kernel_size=sec_k[0]),
            getActivation(actaux),
            
            nn.ReplicationPad1d(sec_k[1]//2),
            nn.Conv1d(in_channels=4,out_channels=8,kernel_size=sec_k[1]),
            getActivation(actaux),
        )
        self.mixer = nn.Sequential(
            nn.Conv1d(in_channels=mdim+8,out_channels=2*mdim,kernel_size=1),
            getActivation(actaux),
            nn.Conv1d(in_channels=2*mdim,out_channels=2*mdim,kernel_size=1),
            getActivation(actaux),
            nn.Conv1d(in_channels=2*mdim,out_channels=mdim,kernel_size=1),
        )
        
    def forward(self,x):
        main = self.main_path(x)
        side = self.sec_path(x)
        x = torch.cat([main,side],dim=-2)
        return self.mixer(x).mT.contiguous()

class BaseEncoder(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        self.eeg_encoder = EEGEncoder()
        self.gsr_encoder = AuxilliaryEncoder(
            main_dils=[2,3,7],
            sec_k=[5,5]
        )
        self.bvp_encoder = AuxilliaryEncoder(
            main_dils=[2,3,5],
            sec_k=[3,3]
        )
        
    def forward(self, eeg, gsr, bvp): return self.eeg_encoder(eeg), self.gsr_encoder(gsr), self.bvp_encoder(bvp)
        
class MambaBlock(nn.Module):
    
    def __init__(self,mdims, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mambaA = Mamba(mdims)
        self.mambaB = Mamba(mdims)
        
        self.normA = nn.InstanceNorm1d(mdims, affine=True)
        self.normB = nn.InstanceNorm1d(mdims, affine=True)
        
        self.wt1 = nn.Parameter(torch.full((1, 1, mdims), 0.25))
        self.wt2 = nn.Parameter(torch.full((1, 1, mdims), 0.25))
        
    def forward(self,x:torch.Tensor):
        
        dupe = x
        x = self.normA(x.mT).mT
        x = self.mambaA(x) * self.wt1 + dupe
        
        dupe = x
        x = self.normB(x.mT).mT
        x = self.mambaB(x) * self.wt2 + dupe
        
        return x

class MambaLayer(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        self.eeg_mamba = MambaBlock(mdims=mdim)
        self.gsr_mamba = MambaBlock(mdims=mdimp)
        self.bvp_mamba = MambaBlock(mdims=mdimp)
    
    def forward(self, eeg, gsr, bvp): return self.eeg_mamba(eeg), self.gsr_mamba(gsr), self.bvp_mamba(bvp)

class AttentionPooling(nn.Module):
    
    def __init__(self, mdims):
        super().__init__()
        self.project = nn.Linear(mdims, mdims//2)
        self.weigh = nn.Linear(mdims//2, 1)
        
    def forward(self,x):
        H = torch.tanh(self.project(x))
        W = torch.softmax(self.weigh(H), dim=-2)
        return torch.sum(W * x, dim=-2)
    
class BaseClassifer(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        self.encoder = BaseEncoder()
        self.mamba = MambaLayer()