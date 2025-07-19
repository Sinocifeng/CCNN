from datetime import datetime
from math import exp
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from model.nodeemb_model import Node_encoder, Node_decoder, SimilarityPredictor, LengthPredictor
from tools.coextent import attribute_co_extent_similarity
from tools.dataset_tool import get_dataset, LatticeDataset, Collate, DataAugment
from tools.visualization_tool import draw_losses

time_str = lambda: datetime.now().strftime("%H:%M:%S.%f")[:-3]


def kl_divergence(mu, logvar):
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

def auc_roc(predicted, target):
    target = target.reshape(-1).cpu().numpy()
    predicted = predicted.reshape(-1).cpu().numpy()
    return roc_auc_score(target, predicted)

def rowwise_weighted_bce_loss(pred_context, context, alpha=0.95,
                              pos_weight=3.0, gamma=1.0, lambda_l1=0.2):
    b, a, _ = context.shape
    valid_nodes = context[:, torch.arange(a), torch.arange(a)].sum(1)
    mask = (torch.arange(a).expand(b, a, a) < valid_nodes.view(-1, 1, 1)).float()

    loss = F.binary_cross_entropy(pred_context, context.float(), reduction='none')
    p_t = context * pred_context + (1 - context) * (1 - pred_context)
    focal_factor = (1 - p_t + 1e-6) ** gamma
    weighted_focal_loss = focal_factor * loss * (context * pos_weight + (1 - context))

    padding_loss = (weighted_focal_loss * (1 - mask)).sum() / ((1 - mask).sum() + 1e-6)
    padding_l1_loss = (pred_context * (1 - mask)).abs().sum() / ((1 - mask).sum() + 1e-6)
    adj_loss = (F.binary_cross_entropy(pred_context, context.float(), reduction='none') * mask).sum() / mask.sum()

    total = (1 - alpha) * padding_loss + alpha * adj_loss + lambda_l1 * padding_l1_loss
    return total


@torch.no_grad()
def dev_performance(epoch):
    losses, kls, co_losses, sz_losses = [], [], [], []
    for ctx, eq, *_ , sz1, _ in train_dataloader:
        ctx = ctx[:, :, :ctx.size(2)].to(device)
        eq, sz1 = eq.to(device), sz1.to(device)

        col_emb, mu, logvar = encoder(ctx)
        node_emb = encoder.encode_rows(ctx, col_emb)
        pred = decoder(col_emb, node_emb)

        loss = rowwise_weighted_bce_loss(pred, ctx.float())
        kl = beta * kl_divergence(mu, logvar)

        losses.append(loss.item())
        kls.append(kl.item())

        if mode_metric_learn:
            sim_pred = similarity_predictor(node_emb[:, :, att_coextent_slice])
            co_loss = mse_criterion(sim_pred, attribute_co_extent_similarity(eq))
            co_losses.append(co_loss.item())

            sz_loss = mse_criterion(length_predictor(node_emb[:, :, sizes_slice]), sz1.float())
            sz_losses.append(sz_loss.item())

    log = {"set": "dev", "epoch": epoch,
           "reconstruction loss": sum(losses) / len(losses),
           "kl divergence": sum(kls) / len(kls),
           "kl annealing": annealing(epoch)}
    if mode_metric_learn:
        log.update({
            "metric weight": metric_weight(epoch),
            "co-extent metric loss": sum(co_losses) / len(co_losses),
            "number of concept metric loss": sum(sz_losses) / len(sz_losses)
        })

    msg = (f"{time_str()} dev: {log['reconstruction loss']:9.5f}, "
           f"kl: {log['kl divergence']:9.5f} x {log['kl annealing']:.3f}")
    if mode_metric_learn:
        msg += (f", co-extent loss: {log['co-extent metric loss']:11.5f} "
                f"x {log['metric weight']:.3f}")
        msg += (f", number of concept loss: {log['number of concept metric loss']:11.5f} "
                f"x {log['metric weight']:.3f}")
    print(msg)
    return log


if __name__ == '__main__':
    mode_metric_learn = False
    module_name = "equi_vae"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size, shuffle, num_workers = 4, True, 4
    lr, weight_decay, epochs = 1e-4, 0, 5

    pre_emb_size, emb_size = 64, 124
    att_coextent_slice = slice(0, emb_size // 2)
    sizes_slice = slice(emb_size // 2, emb_size // 2 + emb_size // 4)

    metric_weight = lambda e: min(e / (epochs / 2), 1)
    beta, slope, center = 1e-3, 2, 30
    annealing = lambda e: 1. / (1. + exp(slope * (-e + center)))

    train_ds, dev_ds = get_dataset("./data/train")
    eval_ds = LatticeDataset("./data/test")

    augment = DataAugment(0.7)
    collate = Collate(augment)

    train_loader = DataLoader(train_ds, batch_size, shuffle, num_workers=num_workers,
                              collate_fn=collate, pin_memory=True)
    dev_loader   = DataLoader(dev_ds,   batch_size, shuffle, num_workers=num_workers,
                              collate_fn=collate, pin_memory=True)
    eval_loader  = DataLoader(eval_ds,  batch_size, shuffle, num_workers=num_workers,
                              collate_fn=collate, pin_memory=True)

    if not mode_metric_learn:
        encoder = Node_encoder(emb_size=emb_size, pre_emb_size=pre_emb_size).to(device)
        decoder = Node_decoder(emb_size=emb_size).to(device)
        module_data = {"encoder": encoder, "decoder": decoder}
    else:
        module_data = torch.load(f"module/{module_name}.tch", map_location=device)
        encoder, decoder = module_data["encoder"], module_data["decoder"]
        similarity_predictor = SimilarityPredictor(emb_size // 2).to(device)
        length_predictor     = LengthPredictor(emb_size // 4).to(device)
        module_data.update({
            "similarity_predictor": similarity_predictor,
            "length_predictor": length_predictor,
            "node_coextent_slice": att_coextent_slice,
            "sizes_slice": sizes_slice
        })
        module_name += '_metric'


    bce_criterion = nn.BCELoss()
    mse_criterion = nn.MSELoss()
    optim = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()),
                             lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim, patience=20, verbose=True, eps=1e-7, factor=0.1, cooldown=20)

    if mode_metric_learn:
        optim_sim = torch.optim.Adam(
            list(encoder.parameters()) + list(decoder.parameters()) + list(similarity_predictor.parameters()), lr)
        optim_len = torch.optim.Adam(
            list(encoder.parameters()) + list(decoder.parameters()) + list(length_predictor.parameters()), lr)

    logs, epoch_loss = [], []
    for epoch in range(epochs):
        print(f"第 {epoch + 1} / {epochs} 轮训练")
        ctx_means, pred_means = [], []
        losses, kls, co_losses, sz_losses = [], [], [], []

        for ctx, eq, *_ , sz1, _ in train_loader:
            ctx = ctx[:, :, :ctx.size(2)].to(device)
            eq, sz1 = eq.to(device), sz1.to(device)

            col_emb, mu, logvar = encoder(ctx)
            node_emb = encoder.encode_rows(ctx, col_emb)
            pred = decoder(col_emb, node_emb)

            ctx_means.append(ctx.mean().item())
            pred_means.append(pred.mean().item())

            loss = rowwise_weighted_bce_loss(pred, ctx.float())
            kl = beta * kl_divergence(mu, logvar)
            loss += kl * beta * annealing(epoch)

            if mode_metric_learn:
                if epoch % 2 == 0:
                    co_loss = mse_criterion(
                        similarity_predictor(node_emb[:, :, att_coextent_slice]),
                        attribute_co_extent_similarity(eq))
                    co_losses.append(co_loss.item())
                    optim_sim.zero_grad()
                    loss.backward()
                    optim_sim.step()
                else:
                    sz_loss = mse_criterion(
                        length_predictor(node_emb[:, :, sizes_slice]), sz1.float())
                    sz_losses.append(sz_loss.item())
                    loss += sz_loss * metric_weight(epoch)
                    optim_len.zero_grad()
                    loss.backward()
                    optim_len.step()
            else:
                optim.zero_grad()
                loss.backward()
                optim.step()

            losses.append(loss.item())
            kls.append(kl.item())

        print(f"Mean context: {sum(ctx_means)/len(ctx_means):.4f}  "
              f"Mean pred_context: {sum(pred_means)/len(pred_means):.4f}")

        if epoch > center:
            scheduler.step(sum(losses) / len(losses) + sum(kls) / len(kls))

        log = {"set": "train", "epoch": epoch,
               "reconstruction loss": sum(losses) / len(losses),
               "kl divergence": sum(kls) / len(kls),
               "kl annealing": annealing(epoch)}
        if mode_metric_learn:
            log.update({
                "metric weight": metric_weight(epoch),
                "co-extent metric loss": sum(co_losses) / max(len(co_losses), 1),
                "number of concept metric loss": sum(sz_losses) / max(len(sz_losses), 1)
            })

        logs.append(log)
        logs.append(dev_performance(epoch))
        module_data["epoch logs"] = logs
        torch.save(module_data, f"module/{module_name}.tch")

    draw_losses(epoch_loss, "metric_learn" if mode_metric_learn else "train")

    aucs = []
    for ctx, *_ in eval_loader:
        ctx = ctx[:, :, :ctx.size(2)].to(device)
        with torch.no_grad():
            col_emb, _, _ = encoder(ctx)
            node_emb = encoder.encode_rows(ctx, col_emb)
            pred = decoder(col_emb, node_emb)
            aucs.append(auc_roc(pred, ctx))
    print(f"测试集 AUC-ROC = {sum(aucs) / len(aucs)}")