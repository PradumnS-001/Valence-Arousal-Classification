import torch
from torch.utils.data import Dataset, DataLoader
import os
import pickle
from typing import TypedDict
import numpy

from extraUtils.misc import get_leaf_files
from configs import *

class DEAPDict(TypedDict):
    patient_label: torch.Tensor
    emotion_label: torch.Tensor
    eeg: torch.Tensor
    gsr: torch.Tensor
    bvp: torch.Tensor

class DEAP(Dataset):
    
    def __init__(self, files:list[str],mode:str='train',p:float=0.3):
        self.files = files
        self.mode = mode
        self.p = p
        
        self.weights = [1e-6]*4
        data_list = []
        label_list = []
        
        for file in self.files:
            with open(file, 'rb') as f:
                subject = pickle.load(f, encoding='latin1')
                data_list.append(subject['data'])
                label_list.append(subject['labels'])
                
        self.data = numpy.concatenate(data_list, axis=0)
        self.labels = numpy.concatenate(label_list, axis=0)
        
        self.mapped_labels = numpy.zeros(len(self.labels), dtype=int)
        valence = self.labels[:, 0]
        arousal = self.labels[:, 1]
        
        self.mapped_labels[(valence < 5.0) & (arousal < 5.0)] = 0
        self.mapped_labels[(valence < 5.0) & (arousal >= 5.0)] = 1
        self.mapped_labels[(valence >= 5.0) & (arousal < 5.0)] = 2
        self.mapped_labels[(valence >= 5.0) & (arousal >= 5.0)] = 3
        
        unique_classes, counts = numpy.unique(self.mapped_labels, return_counts=True)
        for cls, count in zip(unique_classes, counts):
            self.weights[cls] += count
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index)->DEAPDict:
        
        eeg = torch.tensor(self.data[index, :32, 23*128:59*128], dtype=torch.float32)
        gsr = torch.tensor(self.data[index, 36, 23*128:], dtype=torch.float32)
        bvp = torch.tensor(self.data[index, 38, 23*128:], dtype=torch.float32)
        label = self.mapped_labels[index]
        
        eeg = eeg * (eeg.var(dim=-1, keepdim=True)>1e-5).float()
        gsr = gsr * (gsr.var(dim=-1, keepdim=True)>1e-5).float()
        bvp = bvp * (bvp.var(dim=-1, keepdim=True)>1e-5).float()
        
        if self.mode == 'train':
            ran = numpy.random.uniform()
            if ran <= self.p / 2 and bvp.abs().sum(): gsr = torch.zeros_like(gsr)
            if ran > self.p / 2 and ran <= self.p and gsr.abs().sum(): bvp = torch.zeros_like(bvp)
        
        return {
            'patient_label':torch.tensor(index // 40, dtype=torch.long),
            'emotion_label':torch.tensor(label, dtype=torch.long),
            'eeg':(eeg - eeg.mean(dim=-1, keepdim=True)) / (eeg.std(dim=-1, keepdim=True) + 1e-6),
            'gsr':(gsr - gsr.mean(dim=-1, keepdim=True)) / (gsr.std(dim=-1, keepdim=True) + 1e-6),
            'bvp':(bvp - bvp.mean(dim=-1, keepdim=True)) / (bvp.std(dim=-1, keepdim=True) + 1e-6)
            }
        
def build_loaders(dataset:str='DEAP')->tuple[DataLoader[DEAPDict], DataLoader[DEAPDict]]:
    
    dataset = dataset.upper()
    val_wts = []
    trn_wts = []
    
    if dataset == 'DEAP':
        
        _, files = get_leaf_files(
            path=os.path.join('Data', 'DEAP'),
            ender='.dat'
        )
        numpy.random.shuffle(files)
        
        val = DEAP(files=[files[-1]],mode='val')
        val_wts = sum(val.weights) / numpy.array(val.weights)
        
        trn = DEAP(files=files[:-1],mode='train')
        trn_wts = sum(trn.weights) / numpy.array(trn.weights)
        
    trnLoader = DataLoader(dataset=trn, batch_size=32, shuffle=True,pin_memory=True,num_workers=4, persistent_workers=True, prefetch_factor=2)
    valLoader = DataLoader(dataset=val, batch_size=32, shuffle=False,pin_memory=True,num_workers=4, persistent_workers=True, prefetch_factor=2)
    
    return trnLoader, trn_wts / (numpy.min(trn_wts) + 1e-6), valLoader, val_wts / (numpy.min(val_wts) + 1e-6)
    
if __name__ == '__main__':
    
    _,tw,data,vw = build_loaders(dataset='deap')
    print(tw)
    print(vw)
    for mapper in data:
        for key, val in mapper.items():
            print(key, val.shape, val[0])
        break