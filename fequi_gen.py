import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader
import pandas as pd

from model.fequi_module import FairEquiGenerator
from tools.dataset_tool import get_dataset, LatticeDataset, Collate
from datetime import datetime

time = lambda: datetime.now().strftime("%H:%M:%S.%f")[:-3]

def clip_grad(max_norm=1e6):
    nn.utils.clip_grad.clip_grad_norm_(fequi_generator.parameters(), max_norm)

def find_optimal_threshold(predicted, target):
    target = target.reshape(-1).cpu().numpy()
    predicted = predicted.reshape(-1).cpu().numpy()

    fpr, tpr, threshold = roc_curve(target, predicted)
    i = np.arange(len(tpr))
    roc = pd.DataFrame({'tf': pd.Series(tpr - fpr, index=i),
                        'threshold': pd.Series(threshold, index=i)})
    roc_t = roc.iloc[roc.tf.idxmax()]

    return roc_t['threshold']


def auc_roc(predicted, target):
    target = target.reshape(-1).cpu().numpy()
    predicted = predicted.reshape(-1).cpu().numpy()
    return roc_auc_score(target, predicted)

def accuracy_discovery(pred_fequi: torch.Tensor, truth_fequi: torch.Tensor, new_threshold):  # size = [batch x max_fc x max_node]
    if len(pred_fequi.size()) == 2:


        true_res = torch.sum((pred_fequi >= new_threshold) & (truth_fequi == 1) | (pred_fequi < new_threshold) & (truth_fequi == 0)).item()
        false_res = torch.sum((pred_fequi >= new_threshold) & (truth_fequi == 0) | (pred_fequi < new_threshold) & (truth_fequi == 1)).item()

        accuracy = (true_res) / (true_res + false_res)
        return {"accuracy":accuracy}
    else:
        return [accuracy_discovery(pred_fequi[i], truth_fequi[i], new_threshold) for i in
                range(pred_fequi.size(0))]

def Myloss(input, target, threshold):
    b, n, m = input.shape
    all_zero_mask = (target.sum(dim=2) == 0)
    non_zero_vector_counts = (~all_zero_mask).sum(dim=1, keepdim=True).float()  # [b, 1]

    cropped_inputs = torch.zeros((b, n, m), device=input.device)
    for i in range(b):
        count = int(non_zero_vector_counts[i].item())
        cropped_inputs[i, :count] = input[i, :count]

    input = torch.sigmoid(cropped_inputs)
    error = input - target
    abs_error = torch.abs(error)

    dis = input - threshold
    mask = error <= 0

    abs_error[mask] *= 10

    return abs_error.mean()



def dev_performance(epoch, losses):
    records = []
    dev_losses = []
    for attr_contexts, _, fequiconcepts, _, stops2, _, sizes2 in dev_dataloader:
        contexts = attr_contexts[:, :, :len(attr_contexts[0])].to(device)
        attrs = attr_contexts[:, :, len(attr_contexts[0]):].to(device)  
        fequiconcepts = fequiconcepts.to(device)
        stops2 = stops2.to(device)

        # compute embeddings
        with torch.no_grad():
            node_emb, mu, logvar = equi_module['encoder'](contexts)
            pred_size = equi_module['length_predictor'](node_emb[:, :, equi_module["sizes_slice"]])
            predicted_size = fequi_num_predictor(node_emb, pred_size, attrs)
            pred_fequi = fequi_generator(node_emb, predicted_size, attrs, stops2.size(1))


            loss = Myloss(pred_fequi, fequiconcepts, threshold)      

            dev_losses.append(loss.cpu().item())

        try:
            pred_fequi = torch.sigmoid(pred_fequi)
            new_threshold = find_optimal_threshold(pred_fequi.cpu(), fequiconcepts.cpu())

            auc = auc_roc(pred_fequi.cpu(), fequiconcepts.cpu())
            records += [{**scores, 'threshold': new_threshold, 'AUC ROC': auc, 'failure rate': 0} for scores in
                            accuracy_discovery(pred_fequi.cpu() > new_threshold, fequiconcepts.cpu(), new_threshold)]
        except ValueError:
            records += [{'failure rate': 1}]

    print(len(dev_losses), len(losses))
    if epoch <= 0:
        print(f"{time()}: start training, dev: {sum(dev_losses) / len(dev_losses):7.3f}")
    else:
        print(
            f"{time()} [{epoch + 1}/{epochs}]: train: {sum(losses) / len(losses):7.3f}, dev: {sum(dev_losses) / len(dev_losses):7.3f}")

    print(f'dev : pred_fequi is {pred_fequi[0, 1, :]}\n'
          f'truth: fequiconcepts is {fequiconcepts[0, 1, :]}')
    print("fairConcepts")
    df = pd.DataFrame.from_records(records)
    print(df.mean().apply(lambda x: f'{x:.3f}') + " +- " + df.std().apply(lambda x: f'{x:.3f}'))

    return new_threshold


if __name__ == '__main__':
    # 超参数
    device = "cuda" if torch.cuda.is_available() else "cpu"
    threshold = 0.5
    # BoA
    emb_size = 124
    # LSTM
    attribute_summary_size = 64
    hidden_size = 128
    heads = 4
    layers = 4
    batch_size = 4
    shuffle = True
    num_workers = 4
    lr = 1e-4#2
    dropout = 0.25
    epochs = 10 #300
    betas = 0.9, 0.999

    # 数据集
    train_dataset, dev_dataset = get_dataset("data/train")
    eval_dataset = LatticeDataset("data/test")
    equi_module = torch.load("./module/node_metric.tch", map_location=device)             
    fequi_num_predictor = torch.load("./module/fc_num_bound.tch", map_location=device)           
    collate = Collate()

    train_dataloader = DataLoader(train_dataset,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            num_workers=num_workers,
                            collate_fn=collate)
    dev_dataloader = DataLoader(dev_dataset,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            num_workers=num_workers,
                            collate_fn=collate)
    eval_dataloader = DataLoader(eval_dataset,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            num_workers=num_workers,
                            collate_fn=collate)

    # reset random seed
    torch.manual_seed(0)
    fequi_generator = FairEquiGenerator(emb_size, attribute_summary_size, hidden_size, num_layers=layers, heads=heads, dropout=dropout).to(device)

    def prod(iterable):
        p = 1
        for x in iterable:
            p *= x
        return p
    print("概念模型参数规模：", sum(prod(param.size()) for param in fequi_generator.parameters()))

    # mse = nn.MSELoss()
    bce_criterion = nn.BCEWithLogitsLoss()
    # l1 = nn.L1Loss()

    optim = torch.optim.Adam([
        {'params': fequi_generator.parameters()},
    ], lr, betas=betas)

    # train
    for epoch in range(epochs):
        losses = []
        for attr_contexts, _, fequiconcepts, _, stops2, _, sizes2 in train_dataloader:
            contexts = attr_contexts[:, :, :len(attr_contexts[0])].to(device)
            attrs = attr_contexts[:, :, len(attr_contexts[0]):].to(device)  # attrs = [batch x num_node x num_attr] num_attr=10
            fequiconcepts = fequiconcepts.to(device)
            stops2 = stops2.to(device)

            with torch.no_grad():
                node_emb, mu, logvar = equi_module['encoder'](contexts)
                pred_size = equi_module['length_predictor'](node_emb[:, :, equi_module["sizes_slice"]])
                predicted_size = fequi_num_predictor(node_emb, pred_size, attrs)


            pred_fequi = fequi_generator(node_emb, predicted_size, attrs, stops2.size(1))
            loss = Myloss(pred_fequi, fequiconcepts, threshold)
            losses.append(loss.cpu().item())
            optim.zero_grad()
            loss.backward()
            clip_grad()
            optim.step()
            break
        log = {"set": "fequitrain", "epoch": epoch}
        log["reconstruction loss"] = sum(losses) / len(losses)
        print(f"{time()} {epoch:3}: {log['reconstruction loss']:9.5f}")

        threshold = dev_performance(epoch, losses)
        print('the threshold is :' , threshold)
        torch.save({
            'fequi_generator': fequi_generator,
            'optim': optim,
            'threshold':threshold}, "module/fequiconcept.tch")

    print('threshold is ', threshold)

