import os
import math
import time
import pickle
import numpy as np
import torch
import tqdm
import model.models as models
import model.loaders as loaders
import model.utils as utils

class Trainer(object):
    def __init__(self, model, loader, **kwargs):
        self.params = kwargs
        self.num_epochs = self.params["num_epochs"]
        self.early_stop_min_delta = self.params["early_stop_min_delta"]
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
        os.makedirs(self.output_dir)
        with open(os.path.join(self.output_dir, "params.pickle"), "wb") as f:
            pickle.dump(self.params, f)

    def _loss_fn(self, pred, data):
        raise NotImplementedError

    def _calc_metrics_train(self, pred, data):
        return {}

    def _calc_metrics_eval(self, pred, data):
        return {}
    
    def _train(self):
        self.optimizer.zero_grad()
        batches = self.loader.train_sampler
        batch_records = {}
        t_iter = tqdm.tqdm(batches, desc="\tLoss: ---")
        time_start = time.time() - self.time_ref

        for data in batches:
            # print(utils.torch_mem_usage()) ####
            data.to(self.device)
            # print(utils.torch_mem_usage()) ####
            pred = self.model(data)
            loss = self._loss_fn(pred, data)

            loss.backward()  
            self.optimizer.step()
            
            batch_records.setdefault("loss", []).append(loss.item())
            metrics = self._calc_metrics_train(pred, data)
            for k, v in metrics.items():
                batch_records.setdefault(k, []).append(v)

            t_iter.set_description(f"\tLoss: {loss.item():6.4}")

        time_end = time.time() - self.time_ref

        records = {}
        for k, v in batch_records.items():
            records[k] = np.mean(v)

        records.update[{"time_start": time_start, "time_end": time_end}]

        return records

    @torch.no_grad()
    def _val(self):
        batches = self.loader.train_sampler
        batch_records = {}
        t_iter = tqdm.tqdm(batches, desc="\tLoss: ---")

        for data in batches:
            data.to(self.device)
            pred = self.model(data)
            loss = self._loss_fn(pred, data)

            batch_records.setdefault("loss", []).append(loss.item())
            metrics = self._calc_metrics_val(pred, data)
            for k, v in metrics.items():
                batch_records.setdefault(k, []).append(v)

            t_iter.set_description(f"\tLoss: {loss.item():6.4}")

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
            for i in range(self.num_epochs):
                if torch.cuda.is_available:
                    torch.cuda.empty_cache()

                records_train = self._train()
                records_train.update({"split": "train", "epoch": i, "experiment": self.exp_id})
                self.stats["train"].append(records_train)
                train_epoch_loss = records_train["loss"]
                
                print(f"Train epoch {epoch:04}: average loss = {train_epoch_loss:6.10}")

                records_val = self._val()
                records_val.update({"split": "val", "epoch": i, "experiment": self.exp_id})
                val_epoch_loss = records_val["loss"]
                self.stats["val"].append(records_val)

                print(f"Validation epoch {epoch:04}: average loss = {val_epoch_loss:6.10}")

                savepath = os.path.join(self.params["output_dir"], "model", f"ckpt_epoch_{epoch:04}.pt")
                utils.save_model(model, savepath)

                if val_epoch_loss < best_val_epoch_loss:
                    self.best_val_epoch_loss = val_epoch_loss
                    self.best_model_epoch = epoch

                if np.isnan(train_epoch_loss) and np.isnan(val_epoch_loss):
                    break

                if len(val_epoch_loss_hist) < early_stop_hist_len + 1:
                    val_epoch_loss_hist = [val_epoch_loss] + val_epoch_loss_hist
                else:
                    val_epoch_loss_hist = [val_epoch_loss] + val_epoch_loss_hist[:-1]
                    best_delta = np.max(np.diff(val_epoch_loss_hist))
                    if best_delta < self.early_stop_min_delta:
                        break  

        finally:
            self._save_stats()


class SupTrainer(Trainer):
    def _loss_fn(self, pred, data):
        min_ldist = self.params["min_ldist"]

        pdists = pred["dists"]
        means = pdists[:,:,0]
        lvars = pdists[:,:,1]

        l = (data.pos[data.cell_mask])
        num_cells = l.shape[0]
        rtile = l.unsqueeze(0).expand(num_cells, -1, -1)
        ctile = l.unsqueeze(1).expand(-1, num_cells, -1)
        ldists = torch.clamp(((rtile - ctile)**2).mean(dim=2).log(), min=min_ldist)

        print(means.shape, lvars.shape, ldists.shape) ####
        nll = ((means - ldists) / lvars.exp())**2 / 2 + lvars
        w = data.node_norm[data.cell_mask].sqrt()
        weights = torch.outer(w, w)

        loss = torch.mean(nll * weights)

        return loss




