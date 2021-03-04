import numpy as np
import torch

@torch.no_grad()
def gaussian_nll_l(pred, data, params):
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
def mean_mean_l(pred, data, params):
    pdists = pred["dists"]
    means = pdists[:,:,0]
    mean_mean = torch.mean(means)

    return mean_mean.item()

@torch.no_grad()
def mean_std_l(pred, data, params):
    pdists = pred["dists"]
    lvars = pdists[:,:,1]
    std = torch.exp(lvars / 2)
    mean_std = torch.mean(std)

    return mean_std.item()

@torch.no_grad()
def mse(pred, data, params, lbound=0., ubound=np.inf):
    pdists = pred["dists"]

    idx = (data.dists >= lbound) & (data.dists <= ubound)
    err = ((pdists - data.dists)**2)[idx]

    metric = torch.mean(err)

    return metric.item()

@torch.no_grad()
def mse_l(pred, data, params, lbound=0., ubound=np.inf):
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
def mean_chisq_l(pred, data, params):
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

    x_rank = _get_ranks(pdists, device)
    y_rank = _get_ranks(data.dists, device)

    n = x_rank.shape[1]
    upper = 6 * torch.sum((x_rank - y_rank).pow(2), dim=1)
    down = n * (n ** 2 - 1.0)
    rs = 1.0 - (upper / down)

    return torch.mean(rs).item()

@torch.no_grad()
def spearman_l(pred, data, params):
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

def _trilaterate3D(rad, pos):
    # print(rad) ####
    # print(pos) ####
    p1 = pos[0,:]
    p2 = pos[1,:]
    p3 = pos[2,:]
    p4 = pos[3,:]

    r1 = rad[0]
    r2 = rad[1]
    r3 = rad[2]
    r4 = rad[3]

    e_x = (p2 - p1) / np.linalg.norm(p2 - p1)
    i = np.dot(e_x, (p3 - p1))
    e_y = (p3 - p1 - (i * e_x)) / (np.linalg.norm(p3 - p1 - (i * e_x)))
    e_z = np.cross(e_x, e_y)
    d = np.linalg.norm(p2 - p1)
    j = np.dot(e_y, (p3 - p1))
    x = (r1**2 - r2**2 + d**2) / (2 * d)
    y = ((r1**2 - r3**2 + i**2 + j**2 ) / 2 * j) - ((i / j) * x)
    z1 = np.sqrt(r1**2 - x**2 - y**2)
    z2 = -z1
    # print(x, y, z1) ####
    ans1 = p1 + (x * e_x) + (y * e_y) + (z1 * e_z)
    ans2 = p1 + (x * e_x) + (y * e_y) + (z2 * e_z)
    dist1 = np.linalg.norm(p4 - ans1)
    dist2 = np.linalg.norm(p4 - ans2)

    if np.abs(r4 - dist1) < np.abs(r4 - dist2):
        return ans1
    else: 
        return ans2

_trilaterate3D_v = np.vectorize(_trilaterate3D, excluded=[1], signature="(n)->(m)")

@torch.no_grad()
def tril_cons(pred, data, params, num_trials=5):
    pdists = pred["dists"].cpu().detach().numpy() 
    locs = data.pos[data.cell_mask].cpu().detach().numpy()
    ncells = locs.shape[0]
    preds = []
    for _ in range(num_trials):
        sel = np.random.choice(ncells, 4, replace=False)
        rad = pdists[:,sel]
        pos = locs[sel,:]
        pred = _trilaterate3D_v(rad, pos)
        # print(pred) ####
        preds.append(pred)

    preds = np.array(preds)
    preds -= np.nanmean(preds, axis=0, keepdims=True)

    mnorms = np.nanmean(np.sqrt((preds**2).sum(axis=2)))

    return mnorms

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

@torch.no_grad()
def nll_vae_struct(pred, data, params):
    nll = ((pred["coords"] - data.cell_pos)**2).sum(dim=1) / params["vae_struct_nll_std"] / 2
    metric = nll.mean()

    return metric.item()

@torch.no_grad()
def nll_vae_exp(pred, data, params):
    nll = ((pred["exp"] - data.cell_exp)**2).sum(dim=1) / params["vae_exp_nll_std"] / 2
    metric = nll.mean()

    return metric.item()

def _kl(mean_0, std_0, lstd_0, mean_1, std_1, lstd_1):
    kl = (
        (std_0**2 + (mean_1 - mean_0)**2) / (2 * std_1**2) 
        + lstd_1 
        - lstd_0 
        - 0.5
    ).sum(dim=1)

    return kl

@torch.no_grad()
def kl_vae_struct(pred, data, params):
    kl = _kl(pred["emb_mean"], pred["emb_std"], pred["emb_lstd"], 1., 1., 0.)
    metric = kl.mean()

    return metric.item()

@torch.no_grad()
def kl_vae_exp(pred, data, params):
    kl = _kl(pred["aux_enc_mean"], pred["aux_enc_std"], pred["aux_enc_lstd"], pred["emb_mean"], pred["emb_std"], pred["emb_lstd"])
    metric = kl.mean()
    
    return metric.item()
