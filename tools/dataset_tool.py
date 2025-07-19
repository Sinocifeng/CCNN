import pickle

import torch
from torch import nn, tensor
from torch.utils.data import Dataset
from pathlib import Path
import typing as t

def load_latticeData(file: str):

    with open(file, "rb") as fr:
        data = pickle.load(fr)
    attr_context = data['attr_context']
    equiconcepts = data['equiconcepts']
    fair_equiconcepts = data['fair_equiconcepts']
    return attr_context, equiconcepts, fair_equiconcepts


class LatticeDataset(Dataset):

    root: Path
    files: t.List[Path]

    def __init__(self, folder):
        self.root = Path(folder)
        self.files = [file for file in self.root.rglob('*.pkl')]

    def __len__(self) -> int:    
        return len(self.files)

    def __str__(self) -> str:
        return f"{self.__class__.__name__} object of {len(self)} lattices; 位于文件夹位置为： '{self.root}'"

    def __getitem__(self, i):
        attr_context, equiconcepts, fair_equiconcepts = load_latticeData(self.files[i])
        return attr_context, equiconcepts, fair_equiconcepts



def get_dataset(path="dataset/train", train_ratio = 0.9):

    torch.manual_seed(0)
    dataset = LatticeDataset(path)

    train_split = int(len(dataset) * train_ratio)
    train_dataset, dev_dataset = torch.utils.data.random_split(dataset, [train_split, len(dataset) - train_split])
    return train_dataset, dev_dataset

class DataAugment:

    def __init__(self, sub_rate=0.7):
        self.sub_rate = sub_rate

    def shuffle(self, contexts, equics, fair_equics):
       
        num_node = contexts.size()[-2:-1][0]
        num_attr = contexts.size()[-1] - num_node
        num_c = equics.size()[-2:-1][0]
        num_fc = fair_equics.size()[-2:-1][0]
      
        node_perm = torch.randperm(num_node)
        c_perm = torch.randperm(num_c)
        fc_perm = torch.randperm(num_fc)
        if contexts.dim() == 2:
            
            remaining_index = torch.tensor([i for i in range(contexts.shape[1]) if i not in node_perm])
            contexts = contexts[node_perm][:, torch.cat((node_perm, remaining_index[:num_attr]))]
                
            equics = equics[: , node_perm]
            fair_equics = fair_equics[:, node_perm]
            
            equics = equics[c_perm, :]
            fair_equics = fair_equics[fc_perm, :]

            return contexts, equics, fair_equics
        elif contexts.dim() == 3:
            remaining_index = torch.tensor([i for i in range(contexts.shape[2]) if i not in node_perm])
            contexts = contexts[:,node_perm][:, :, torch.cat((node_perm, remaining_index[:num_attr]))]
            equics = equics[:, :, node_perm]
            fair_equics = fair_equics[:, :, node_perm]

            equics = equics[:, c_perm, :]
            fair_equics = fair_equics[:, fc_perm, :]

            return contexts, equics, fair_equics

    def sub(self, contexts, equics, fair_equics):
        # pass
        return contexts, equics, fair_equics

    def __call__(self, contexts, equics, fair_equics):

        contexts, equics, fair_equics = self.shuffle(contexts, equics, fair_equics)
        if self.sub_rate > 0:
            contexts, equics, fair_equics = self.sub(contexts, equics, fair_equics)
        return contexts, equics, fair_equics


def pad_2D_attr(attr_matrix, target_row_num, masks=False):

    missing_nodes = target_row_num - len(attr_matrix)
    split_tensors = torch.split(attr_matrix, split_size_or_sections=len(attr_matrix), dim=1)
    context, attr = split_tensors[:]

    context = nn.functional.pad(context, (0, missing_nodes, 0, missing_nodes), value=False)
    attr = nn.functional.pad(attr, (0, 0, 0, missing_nodes), value=False)
    padded_attr_matrix = torch.cat((context, attr), dim=1)

    if masks:
        mask = torch.zeros(target_row_num, target_row_num + 10, dtype=torch.bool)
        mask[:len(attr_matrix), :len(attr_matrix[0])] = True
        return padded_attr_matrix, mask
    else:
        return padded_attr_matrix

def pad_2D(c_matrix, target_node_num, max_equi, masks=False):

    missing_nodes = target_node_num - len(c_matrix[0])
    missing_equis = max_equi - len(c_matrix)
    padded_c_matrix = nn.functional.pad(c_matrix, (0, missing_nodes, 0, missing_equis), value=False)
    if masks:
        mask = torch.zeros(max_equi, target_node_num, dtype=torch.bool)
        mask[:len(c_matrix), :len(c_matrix[0])] = True
        return padded_c_matrix, mask
    else:
        return padded_c_matrix



class Collate:
    def __init__(self, data_augment=None, masks=False):
        self.masks = masks
        self.data_augment = data_augment

    def __call__(self, batch):

        samples = []
        for sample_attr_context, sample_equiconcepts, sample_fair_equiconcepts in batch:
            sample_context = torch.from_numpy(sample_attr_context).type(torch.float)
            sample_equics = torch.from_numpy(sample_equiconcepts).type(torch.float)
            sample_fairequics = torch.from_numpy(sample_fair_equiconcepts).type(torch.float)

            if self.data_augment is not None:
                sample_context, sample_equics, sample_fairequics = self.data_augment(sample_context, sample_equics, sample_fairequics)

            samples.append((sample_context, sample_equics, sample_fairequics))

        I = []

        max_node = max(len(context) for context, equi, fair_equi in samples)            
        max_equi = max(len(equi) for context, equi, fair_equi in samples)              
        max_fair_equi = max(len(fair_equi) for context, equi, fair_equi in samples)     

        contexts, equiconcepts, fair_equiconcepts = [], [], []
        c_mask, equi_mask, fair_equi_mask = [], [], []
        for sample_context, sample_equics, sample_fairequics in samples:
            if sample_fairequics.numel() == 0:
                sample_fairequics = tensor([[0]])

            sample_context, mask = pad_2D_attr(sample_context, max_node, masks=True)
            contexts.append(sample_context)
            c_mask.append(mask)

            sample_equics, mask = pad_2D(sample_equics, max_node, max_equi, masks=True)
            equiconcepts.append(sample_equics)
            equi_mask.append(mask)

            sample_fairequics, mask = pad_2D(sample_fairequics, max_node, max_fair_equi, masks=True)
            fair_equiconcepts.append(sample_fairequics)
            fair_equi_mask.append(mask)

        sizes1 = torch.tensor([len(sample_equiconcepts) - 1 for _, sample_equiconcepts, sample_fair_equiconcepts in batch])
        sizes2 = torch.tensor([len(sample_fair_equiconcepts) - 1 for _, sample_equiconcepts, sample_fair_equiconcepts in batch])
        arange1 = torch.arange(max_equi)
        arange2 = torch.arange(max_fair_equi)
        aranges1 = torch.stack([arange1] * len(batch))
        aranges2 = torch.stack([arange2] * len(batch))
        stops1 = (aranges1 == torch.stack([sizes1] * max_equi, dim=-1)).type(torch.long)
        stops2 = (aranges2 == torch.stack([sizes2] * max_fair_equi, dim=-1)).type(torch.long)


        contexts = torch.stack(contexts)
        equiconcepts = torch.stack(equiconcepts)
        fair_equiconcepts = torch.stack(fair_equiconcepts)
        if self.masks:
            c_mask = torch.stack(c_mask)
            equi_mask = torch.stack(equi_mask)
            fair_equi_mask = torch.stack(fair_equi_mask)
            return contexts, equiconcepts, fair_equiconcepts, stops1, stops2, sizes1, sizes2, c_mask.bool(), equi_mask.bool(), fair_equi_mask.bool()
        else:
            return contexts, equiconcepts, fair_equiconcepts, stops1, stops2, sizes1, sizes2