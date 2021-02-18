import os
import math
import time
import pickle
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import model.models as models
import model.loaders as loaders
import model.utils as utils
import model.metrics as metrics

class Trainer(object):
    def __init__(self, model, loader, **kwargs):
        self.params = kwargs
        self.num_epochs = self.params["num_epochs"]
        self.early_stop_min_delta = self.params["early_stop_min_delta"]
        self.early_stop_hist_len = self.params["early_stop_hist_len"]
        self.clip_norm = self.params["grad_clip_norm"]
        self.device = self.params["device"]

        self.model = model
        self.model.to(self.device)
        self.loader = loader
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.params["learning_rate"])

        self.stats = {"train": [], "val": []}
        self.best_val_epoch_loss = np.inf
        self.best_model_epoch = None
        self.time_ref = None

        exp_dir = os.path.join(self.params["results_dir"], self.params["name"])
        os.makedirs(exp_dir, exist_ok=True)
        prev_exps = [int(i) for i in os.listdir(exp_dir) if i.isdecimal()]
        if len(prev_exps) == 0:
            prev_exps = [-1]
        self.exp_id = f"{(max(prev_exps) + 1):04}"
        self.output_dir = os.path.join(exp_dir, self.exp_id)
        os.makedirs(os.path.join(self.output_dir, "model"))
        with open(os.path.join(self.output_dir, "params.pickle"), "wb") as f:
            pickle.dump(self.params, f)

    def _calc_obs(self, data):
        raise NotImplementedError

    def _loss_fn(self, pred, data):
        raise NotImplementedError

    def _calc_metrics_train(self, pred, data):
        return {}

    def _calc_metrics_val(self, pred, data):
        return {}
    
    def _train(self):
        self.optimizer.zero_grad()
        batches = self.loader.train_sampler
        batch_records = {}
        t_iter = tqdm.tqdm(batches, desc="\tLoss: ------", ncols=150)
        time_start = time.time() - self.time_ref

        for data in t_iter:
            # print(data.node_norm) ####
            # print(utils.torch_mem_usage()) ####
            data.to(self.device)
            self._calc_obs(data)
            # print(utils.torch_mem_usage()) ####
            pred = self.model(data)
            loss = self._loss_fn(pred, data)

            loss.backward()  
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.clip_norm)
            self.optimizer.step()
            
            batch_records.setdefault("loss", []).append(loss.item())
            out_metrics = self._calc_metrics_train(pred, data)
            for k, v in out_metrics.items():
                batch_records.setdefault(k, []).append(v)

            t_iter.set_description(f"\tLoss: {loss.item():6.4f}")

        time_end = time.time() - self.time_ref

        records = {}
        for k, v in batch_records.items():
            records[k] = np.mean(v)

        records.update({"time_start": time_start, "time_end": time_end})

        return records

    @torch.no_grad()
    def _val(self):
        batches = self.loader.val_sampler
        batch_records = {}
        t_iter = tqdm.tqdm(batches, desc="\tLoss: ------", ncols=150)

        for data in t_iter:
            print(data.node_norm) ####
            data.to(self.device)
            self._calc_obs(data)
            pred = self.model(data)
            loss = self._loss_fn(pred, data)

            batch_records.setdefault("loss", []).append(loss.item())
            out_metrics = self._calc_metrics_val(pred, data)
            for k, v in out_metrics.items():
                batch_records.setdefault(k, []).append(v)

            t_iter.set_description(f"\tLoss: {loss.item():6.4f}")

        records = {}
        for k, v in batch_records.items():
            records[k] = np.mean(v)

        return records

    def _save_stats(self):
        stats_path = os.path.join(self.output_dir, "stats.pickle")
        with open(stats_path, "wb") as f:
            pickle.dump(self.stats, f)

        best_epoch_path = os.path.join(self.output_dir, "best_model_epoch.txt")
        with open(best_epoch_path, "w") as f:
            if self.best_model_epoch is None:
                f.write(f"{self.best_model_epoch}\n")
            else:
                f.write(f"{self.best_model_epoch:04}\n")

    def run(self):
        val_epoch_loss_hist = []
        self.time_ref = time.time()
        try:
            for epoch in range(self.num_epochs):
                if torch.cuda.is_available:
                    torch.cuda.empty_cache()

                records_train = self._train()
                records_train.update({"split": "train", "epoch": epoch, "experiment": self.exp_id})
                self.stats["train"].append(records_train)
                train_epoch_loss = records_train["loss"]
                
                print(f"Train epoch {epoch:04}: average loss = {train_epoch_loss:6.10}")

                records_val = self._val()
                records_val.update({"split": "val", "epoch": epoch, "experiment": self.exp_id})
                val_epoch_loss = records_val["loss"]
                self.stats["val"].append(records_val)

                print(f"Validation epoch {epoch:04}: average loss = {val_epoch_loss:6.10}")
                print("".join(f"{k:>15}:  {v}\n" for k, v in records_val.items()))

                savepath = os.path.join(self.output_dir, "model", f"ckpt_epoch_{epoch:04}.pt")
                utils.save_model(self.model, savepath)

                if val_epoch_loss < self.best_val_epoch_loss:
                    self.best_val_epoch_loss = val_epoch_loss
                    self.best_model_epoch = epoch

                if np.isnan(train_epoch_loss) and np.isnan(val_epoch_loss):
                    break

                if len(val_epoch_loss_hist) < self.early_stop_hist_len + 1:
                    val_epoch_loss_hist = [val_epoch_loss] + val_epoch_loss_hist
                else:
                    val_epoch_loss_hist = [val_epoch_loss] + val_epoch_loss_hist[:-1]
                    best_delta = np.max(np.diff(val_epoch_loss_hist))
                    if best_delta < self.early_stop_min_delta:
                        break  

        finally:
            self._save_stats()


class SupTrainer(Trainer):
    def _calc_obs(self, data):
        min_dist = self.params["min_dist"]
        l = (data.pos[data.cell_mask])
        num_cells = l.shape[0]
        rtile = l.unsqueeze(0).expand(num_cells, -1, -1)
        ctile = l.unsqueeze(1).expand(-1, num_cells, -1)
        data.dists = ((torch.clamp(rtile - ctile, min=min_dist))**2).mean(dim=2).sqrt()
        data.ldists = data.dists.log()

    def _loss_fn(self, pred, data):
        pdists = pred["dists"]
        means = pdists[:,:,0]
        lvars = pdists[:,:,1]

        nll = ((means - data.ldists) / lvars.exp())**2 / 2 + lvars
        # print(means.sum(), lvars.sum()) ####
        # print(nll.sum()) ####
        w = data.node_norm[data.cell_mask].sqrt()
        weights = torch.outer(w, w)

        loss = torch.sum(nll * weights)
        # print(loss) ####

        return loss

    def _calc_metrics_val(self, pred, data):
        out_metrics = {
            "gaussian_nll": metrics.gaussian_nll(pred, data, self.params),
            "mean_pred_mean": metrics.mean_mean(pred, data, self.params),
            "mean_pred_std": metrics.mean_std(pred, data, self.params),
            "mse": metrics.mse(pred, data, self.params),
            "mse_lt_100": metrics.mse(pred, data, self.params, lbound=100.),
            "mse_100_500": metrics.mse(pred, data, self.params, lbound=100., ubound=500.),
            "mse_gt_500": metrics.mse(pred, data, self.params, lbound=500.),
            "mse_log": metrics.mse_log(pred, data, self.params),
            "mean_chisq": metrics.mean_chisq(pred, data, self.params),
            "spearman": metrics.spearman(pred, data, self.params),
        }
        return out_metrics


class SupBinTrainer(Trainer):
    def _calc_obs(self, data):
        min_dist = self.params["min_dist"]
        l = (data.pos[data.cell_mask])
        num_cells = l.shape[0]
        rtile = l.unsqueeze(0).expand(num_cells, -1, -1)
        ctile = l.unsqueeze(1).expand(-1, num_cells, -1)
        dists = ((torch.clamp(rtile - ctile, min=min_dist))**2).mean(dim=2).sqrt()
        data.adjs = (dists <= self.params["adj_thresh"])
        data.padj = data.adjs.float().mean()

    def _loss_fn(self, pred, data):
        logits = pred["logits"]
        lflat = logits.view(-1, 1)

        dflat = data.adjs.float().view(-1, 1)
        pweight = ((1 - data.padj) / data.padj).clamp(max=1e5)

        w = data.node_norm[data.cell_mask].view(-1)

        loss = F.binary_cross_entropy_with_logits(lflat, dflat, pos_weight=pweight)

        return loss

    def _calc_metrics_val(self, pred, data):
        out_metrics = {
            "acc": metrics.acc(pred, data, self.params),
            "f1": metrics.f1(pred, data, self.params),
            "mcc": metrics.mcc(pred, data, self.params),
        }
        return out_metrics


