import torch
from torch import nn
from einops import rearrange
from mamba_ssm import Mamba
from configs import *

def ganin_scheduler(epoch):
    epoch = max(0,epoch - p_lr*epochs + 1)
    return -1*torch.tanh(epoch/2.5).item()

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
    
class Float32PaddingWrapper(nn.Module):
    def __init__(self, padding_layer):
        super().__init__()
        self.padding_layer = padding_layer

    def forward(self, x:torch.Tensor):
        original_dtype = x.dtype
        x = x.to(torch.float32)
        x = self.padding_layer(x)
        return x.to(original_dtype)
    
class SqueezeExciteAttention(nn.Module):
    
    def __init__(self, dim,actvsea='silu', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bottle_neck = nn.Sequential(
            nn.Linear(dim, dim//4),
            getActivation(actvsea),
            nn.Linear(dim//4, dim),
            nn.Sigmoid()
        )
        
    def forward(self, x:torch.Tensor): return x * (self.bottle_neck(x.mean(-1))).unsqueeze(-1)
        
class EEGBandExtractor(nn.Module):
    def __init__(self, k, d, k2,c, inc=None):
        super().__init__()
        inc = c if inc is None else inc
        self.extractor = nn.Sequential(
            Float32PaddingWrapper(nn.ReplicationPad1d((k-1)*d // 2)),
            nn.Conv1d(in_channels=inc,out_channels=4*c, kernel_size=k, dilation=d, groups=c),
            getActivation(actveeg),
            Float32PaddingWrapper(nn.ReplicationPad1d(k2 // 2)),
            nn.Conv1d(in_channels=4*c,out_channels=c, kernel_size=k2, groups=c),
            getActivation(actveeg),
            SqueezeExciteAttention(dim=c)
        )
        
    def forward(self, x): return self.extractor(x)

class EEGEncoder(nn.Module):
    
    def __init__(self, c, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.alpha_band_extractor = EEGBandExtractor(k=5,d=6,k2=7,c=c)
        self.theta_band_extractor = EEGBandExtractor(k=5,d=3,k2=3,c=c)
        self.beta_band_extractor = EEGBandExtractor(k=5,d=2,k2=3,c=c)
        self.gamma_band_extractor = EEGBandExtractor(k=3,d=1,k2=3,c=c)
        
        self.mixer = nn.Sequential(
            nn.Conv1d(in_channels=c*4,out_channels=c*3,kernel_size=1),
            getActivation(actveeg),
            nn.Conv1d(in_channels=c*3,out_channels=c*2,kernel_size=1),
            getActivation(actveeg),
            nn.Conv1d(in_channels=c*2,out_channels=mdim,kernel_size=1),
        )
        
    def forward(self,x):
        
        alpha = self.alpha_band_extractor(x)
        beta = self.beta_band_extractor(x)
        gamma = self.gamma_band_extractor(x)
        theta = self.theta_band_extractor(x)
        feats = torch.cat([alpha,beta,gamma,theta],dim=1)
        return self.mixer(feats).mT.contiguous()

class AuxilliaryEncoder(nn.Module):
    def __init__(self, main_dils, sec_k):
        super().__init__()
        
        self.main_path = nn.Sequential(
            Float32PaddingWrapper(nn.ReplicationPad1d(main_dils[0]*2)),
            nn.Conv1d(in_channels=1,out_channels=mdimp//4,dilation=main_dils[0],kernel_size=5),
            getActivation(actaux),
            
            Float32PaddingWrapper(nn.ReplicationPad1d(main_dils[1]*2)),
            nn.Conv1d(in_channels=mdimp//4,out_channels=mdimp//2,dilation=main_dils[1],kernel_size=5),
            getActivation(actaux),
            
            Float32PaddingWrapper(nn.ReplicationPad1d(main_dils[2]*2)),
            nn.Conv1d(in_channels=mdimp//2,out_channels=mdimp,dilation=main_dils[2],kernel_size=5),
            getActivation(actaux)
        )
        self.sec_path = nn.Sequential(
            Float32PaddingWrapper(nn.ReplicationPad1d(sec_k[0]//2)),
            nn.Conv1d(in_channels=1,out_channels=4,kernel_size=sec_k[0]),
            getActivation(actaux),
            
            Float32PaddingWrapper(nn.ReplicationPad1d(sec_k[1]//2)),
            nn.Conv1d(in_channels=4,out_channels=8,kernel_size=sec_k[1]),
            getActivation(actaux),
        )
        self.mixer = nn.Sequential(
            nn.Conv1d(in_channels=mdimp+8,out_channels=2*mdimp,kernel_size=1),
            getActivation(actaux),
            nn.Conv1d(in_channels=2*mdimp,out_channels=2*mdimp,kernel_size=1),
            getActivation(actaux),
            nn.Conv1d(in_channels=2*mdimp,out_channels=mdimp,kernel_size=1),
        )
        
    def forward(self,x):
        main = self.main_path(x)
        side = self.sec_path(x)
        x = torch.cat([main,side],dim=-2)
        return self.mixer(x).mT.contiguous()

class BaseEncoder(nn.Module):
    
    def __init__(self,c):
        super().__init__()
        
        self.eeg_encoder = EEGEncoder(c=c)
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
    
    def __init__(self, mdim, mdimp):
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
        
    def forward(self,x,sec=None):
        H = torch.tanh(self.project(x))
        W = torch.softmax(self.weigh(H), dim=-2)
        return torch.sum(W * (x if sec is None else sec), dim=-2)
    
class Conditioner(nn.Module):
    
    def __init__(self, D,C,V):
        super().__init__()
        
        self.Q = nn.Linear(D,V)
        self.Ks = nn.Linear(C,V)
        self.pool = AttentionPooling(mdims=V)
        self.norm = nn.InstanceNorm1d(V, affine=True)
        
    def forward(self,query, key_seq):
        
        query = self.Q(query).unsqueeze(-2)
        keys = self.Ks(key_seq)
        return self.pool(query * self.norm(keys.mT).mT,key_seq)
    
class DomainSplitter(nn.Module):
    
    def __init__(self, indim, outdim, hdim, actuda='silu'):
        super().__init__()
        self.feat_extractor = nn.Sequential(
            nn.Linear(in_features=indim,out_features=hdim),
            nn.BatchNorm1d(hdim, affine=True),
            getActivation(actuda),
            nn.Dropout(0.1),
            
            nn.Linear(in_features=hdim,out_features=hdim),
            nn.BatchNorm1d(hdim, affine=True),
            getActivation(actuda),
            nn.Dropout(0.1)
        )
        self.headA = nn.Linear(hdim,outdim)
        self.headB = nn.Linear(hdim,outdim)
        
    def forward(self, x):
        feat = self.feat_extractor(x)
        return self.headA(feat),self.headB(feat)
    
class BaseTailDEAP(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        if do_uda:
            self.split = DomainSplitter(indim=mdim+2*mdimp,hdim=2*mdim,outdim=emote_dim)
            self.classifyA = nn.Sequential(
                nn.Linear(emote_dim,(emote_dim+4)//2),
                getActivation(actm),
                nn.Linear((emote_dim+4)//2,4)
            )
            self.classifyB = nn.Sequential(
                nn.Linear(emote_dim,(emote_dim+num_pats)//2),
                getActivation(actm),
                nn.Linear((emote_dim+num_pats)//2,num_pats)
            )
            
        else:
            self.classifier = nn.Sequential(
            nn.Linear(mdim+2*mdimp,mdim),
            nn.BatchNorm1d(mdim, affine=True),
            getActivation(actm),
            nn.Dropout(0.25),
            
            nn.Linear(mdim,mdim),
            nn.BatchNorm1d(mdim, affine=True),
            getActivation(actm),
            nn.Dropout(0.25),
            
            nn.Linear(mdim,4)
        )
            
    def forward(self, x, epoch):
        
        if do_uda:
            emot,pat = self.split(x)
            tem = pat
            wt = -1
            if self.training and epoch is not None:
                wt = ganin_scheduler(epoch)
            pat = wt * pat + (1-wt) * pat.detach()
            return self.classifyA(emot),self.classifyB(pat),emot,tem
        else: return self.classifier(x)
    
class BaseClassifierDEAP(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        self.encoder = BaseEncoder(c=32)
        self.mamba = MambaLayer(mdim=mdim,mdimp=mdimp)
        
        self.eeg_pooler = AttentionPooling(mdims=mdim)
        self.gsr_pooler = Conditioner(D=mdim,C=mdimp,V=mdimp)
        self.bvp_pooler = Conditioner(D=mdim,C=mdimp,V=mdimp)
        
        self.eeg_norm = nn.InstanceNorm1d(mdim, affine=True)
        self.classifier = BaseTailDEAP()
        
    def forward(self, x, epoch = None):
        
        eeg,gsr,bvp = x
        B = eeg.shape[0]
        
        eeg = rearrange(eeg, 'b k i j -> (b k) i j')
        gsr = rearrange(gsr, 'b k i j -> (b k) i j')
        bvp = rearrange(bvp, 'b k i j -> (b k) i j')
        
        eeg,gsr,bvp = self.encoder(eeg,gsr,bvp)
        eeg,gsr,bvp = self.mamba(eeg,gsr,bvp)
        
        eeg = self.eeg_pooler(self.eeg_norm(eeg.mT).mT)
        gsr = self.gsr_pooler(eeg,gsr)
        bvp = self.bvp_pooler(eeg,bvp)
        
        res = torch.cat([eeg, gsr, bvp], dim=-1)
        res = res.view(B,-1,mdim+2*mdimp).mean(1)
        return self.classifier(res, epoch)