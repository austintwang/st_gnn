@torch.no_grad
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

def _get_ranks(x):
    tmp = x.argsort(dim=1)
    ranks = torch.zeros_like(tmp)
    for i in range(x.shape[0]):
        ranks[i,tmp[i,:]] = torch.arange(len(x))
    return ranks

@torch.no_grad
def spearman(pred, data, params):
    min_dist = params["min_dist"]

    pdists = pred["dists"]
    means = pdists[:,:,0]

    x_rank = _get_ranks(means)
    y_rank = _get_ranks(data.ldists)

    n = num_cells
    upper = 6 * torch.sum((x_rank - y_rank).pow(2), dim=1)
    down = n * (n ** 2 - 1.0)
    rs = 1.0 - (upper / down)

    return torch.mean(rs).item()
