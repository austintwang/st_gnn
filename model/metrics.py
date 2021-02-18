import numpy as np
import torch

@torch.no_grad()
def gaussian_nll(pred, data, params):
    min_dist = params["min_dist"]

    pdists = pred["dists"]
    means = pdists[:,:,0]
    lvars = pdists[:,:,1]

    nll = ((means - data.ldists) / lvars.exp())**2 / 2 + lvars
    w = data.node_norm[data.cell_mask].sqrt()
    weights = torch.outer(w, w)

    metric = torch.mean(nll * weights)

    return metric.item()

@torch.no_grad()
def mean_mean(pred, data, params):
    pdists = pred["dists"]
    means = pdists[:,:,0]
    mean_mean = torch.mean(means)

    return mean_mean.item()

@torch.no_grad()
def mean_std(pred, data, params):
    pdists = pred["dists"]
    lvars = pdists[:,:,1]
    std = torch.exp(lvars / 2)
    mean_std = torch.mean(std)

    return mean_std.item()

@torch.no_grad()
def mse(pred, data, params, lbound=0., ubound=np.inf):
    pdists = pred["dists"]
    means = pdists[:,:,0]
    emeans = torch.exp(means)

    idx = (data.dists >= lbound) & (data.dists <= ubound)
    err = ((emeans - data.dists)**2)[idx]

    metric = torch.mean(err)

    return metric.item()

@torch.no_grad()
def mse_log(pred, data, params):
    pdists = pred["dists"]
    means = pdists[:,:,0]

    metric = torch.mean((means - data.ldists)**2)

    return metric.item()

@torch.no_grad()
def mean_chisq(pred, data, params):
    pdists = pred["dists"]
    means = pdists[:,:,0]
    lvars = pdists[:,:,1]
    std = torch.exp(lvars / 2)

    chisq = ((data.ldists - means) / std)**2
    metric = torch.mean(chisq)

    return metric.item()

def _get_ranks(x, device):
    tmp = x.argsort(dim=1)
    ranks = torch.zeros_like(tmp, device=device)
    for i in range(x.shape[0]):
        ranks[i,tmp[i,:]] = torch.arange(len(x), device=device)
    return ranks

@torch.no_grad()
def spearman(pred, data, params):
    min_dist = params["min_dist"]
    device = params["device"]

    pdists = pred["dists"]
    means = pdists[:,:,0]

    x_rank = _get_ranks(means, device)
    y_rank = _get_ranks(data.ldists, device)

    n = x_rank.shape[1]
    upper = 6 * torch.sum((x_rank - y_rank).pow(2), dim=1)
    down = n * (n ** 2 - 1.0)
    rs = 1.0 - (upper / down)

    return torch.mean(rs).item()

@torch.no_grad()
def acc(pred, data, params):
    logits = pred["logits"]
    bin_preds = (logits >= 0)

    acc = (bin_preds == data.adjs)

    metric = acc.float().mean()

    return metric.item()

@torch.no_grad()
def f1(pred, data, params):
    logits = pred["logits"]
    bin_preds = (logits >= 0)

    tp = (bin_preds & data.adjs).float().sum()
    fp = (bin_preds & ~data.adjs).float().sum()
    fn = (~bin_preds & data.adjs).float().sum()

    metric = tp / (tp + (fp + fn) / 2)

    return metric.item()

@torch.no_grad()
def mcc(pred, data, params):
    logits = pred["logits"]
    bin_preds = (logits >= 0)

    tp = (bin_preds & data.adjs).float().sum()
    tn = (~(bin_preds | data.adjs)).float().sum()
    fp = (bin_preds & ~data.adjs).float().sum()
    fn = (~bin_preds & data.adjs).float().sum()

    metric = (
        (tp * tn - fp * fn) 
        / torch.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    )

    return metric.item()
