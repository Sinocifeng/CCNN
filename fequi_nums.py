import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from datetime import datetime

from model.fequi_num_model import NumberUpperBoundPredictor
from tools.dataset_tool import get_dataset, LatticeDataset, DataAugment, Collate

time_str = lambda: datetime.now().strftime("%H:%M:%S.%f")[:-3]

def length_loss(input, target):
    error = input - target * 1.1
    error[error < 0] *= 10
    return error.pow(2).mean()

def load_data(path, batch_size, collate_fn, shuffle=True, num_workers=8):
    ds = LatticeDataset(path) if 'test' in path else get_dataset(path)[0]
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, collate_fn=collate_fn,
                      pin_memory=True)

def extract_inputs(batch):
    attr_ctx, equi, fair_equi, _, _, _, sizes2 = batch
    ctx = attr_ctx[:, :, :attr_ctx.size(2)].to(device)
    attrs = attr_ctx[:, :, attr_ctx.size(2):].to(device)
    return ctx, attrs, sizes2.to(device)

@torch.no_grad()
def forward_once(ctx, attrs, sizes):
    node_emb, _, _ = equi_module['encoder'](ctx)
    pred_size = equi_module['length_predictor'](node_emb[:, :, equi_module["sizes_slice"]])
    pred_num = predictor(node_emb, pred_size, attrs)
    return pred_num, sizes.float()

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size, num_workers, lr, epochs = 8, 8, 1e-4, 80
    attr_embsize, emb_size = 10, 124

    collate = Collate(DataAugment(0.1), masks=False)
    train_loader = load_data("./dataset/train", batch_size, collate)
    dev_loader   = load_data("./dataset/train", batch_size, collate)
    eval_loader  = load_data("./dataset/test",  batch_size, collate, shuffle=False)

    torch.manual_seed(0)
    equi_module = torch.load("./module/equi_vae_metric.tch", map_location=device)
    predictor = NumberUpperBoundPredictor(emb_size).to(device)
    optim = torch.optim.Adam(predictor.parameters(), lr)
    print("模型参数数量：", sum(p.numel() for p in predictor.parameters()))

    for epoch in range(epochs):
        losses, dev_losses = [], []

        for batch in train_loader:
            ctx, attrs, sizes = extract_inputs(batch)
            pred_num, target = forward_once(ctx, attrs, sizes)
            loss = length_loss(pred_num, target)
            losses.append(loss.item())

            optim.zero_grad()
            loss.backward()
            optim.step()

        for batch in dev_loader:
            ctx, attrs, sizes = extract_inputs(batch)
            pred_num, target = forward_once(ctx, attrs, sizes)
            dev_losses.append(length_loss(pred_num, target).item())

        print(f"{time_str()} [{epoch+1}/{epochs}]: "
              f"train: {sum(losses)/len(losses):.3f}, "
              f"dev:   {sum(dev_losses)/len(dev_losses):.3f}")

    torch.save(predictor.cpu(), "module/fc_num_bound.tch")
    predictor = predictor.to(device)

    plt.figure()
    for batch in eval_loader:
        ctx, attrs, sizes = extract_inputs(batch)
        pred_num, _ = forward_once(ctx, attrs, sizes)
        plt.scatter(sizes.cpu(), pred_num.cpu(), alpha=0.1, color='blue')

    plt.plot([1, 150], [1, 150], color='orange', label='perfect predictor (y=x)')
    plt.xlabel("true number of concepts")
    plt.ylabel("predicted number of concepts")
    plt.legend()
    plt.savefig("concept_upper_bound.png")