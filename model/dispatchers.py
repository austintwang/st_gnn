import os
import model.models as models
import model.loaders as loaders
import model.trainers as trainers

class Dispatcher(object):
    names = {}
    def __init__(self, name, paramlist, loader_cls, model_cls, trainer_cls):
        Dispatcher.names[name] = self
        self.name = name

        self.params = {}
        for i in paramlist:
            self.params.update(i)

        self.loader_cls = loader_cls
        self.model_cls = model_cls
        self.trainer_cls = trainer_cls

    @classmethod
    def variant(cls, other, name, paramlist):
        paramlist = [other.params] + paramlist
        return cls(name, paramlist, other.loader_cls, other.model_cls, other.trainer_cls)

    def dispatch(self, device, clear_cache):
        self.params.update({"name": self.name, "device": device, "clear_cache": clear_cache})

        loader = self.loader_cls(**self.params)
        model = self.model_cls(loader.node_in_channels, **self.params)
        trainer = self.trainer_cls(model, loader, **self.params)
        trainer.run()


global_params = {
    "num_workers": 8,
}
        
train_params = {
    "num_epochs": 500,
    "learning_rate": 1e-3,
    "early_stop_min_delta": 0.001,
    "early_stop_hist_len": 10,
    "dropout_prop": 0.1,
    "dist_layers_out_chnls": [64],
    "min_dist": 1e-4,
    "grad_clip_norm": 1.,
    "results_dir": "/dfs/user/atwang/data/analyses/st_gnn"
}

gnn_params = {
    "gnn_layers_out_chnls": [256, 256]
}

loader_params = {
    "batch_size": 500,
    "train_prop": 0.8,
}

data_path = "/dfs/user/atwang/data/spt_zhuang/"
coords_path = os.path.join(data_path, "parsed", "cell_coords.pickle")
orgs_path = os.path.join(data_path, "parsed", "cell_orgs.pickle")
exp_path = os.path.join(data_path, "source/processed_data/counts.h5ad")
zhuang_params = {
    "st_exp_path": exp_path,
    "st_coords_path": coords_path,
    "st_organisms_path": orgs_path,
    "st_exp_threshold": 0.001,
}

saint_params = {
    "saint_walk_length": 2,
    "saint_num_steps": {"train": 500, "val": 500},
    "saint_sample_coverage": 100,
    "loader_cache_dir": "/dfs/user/atwang/data/spt_zhuang/cache/saint"
}

test_params = {
    "saint_sample_coverage": 2, 
    "loader_cache_dir": "/dfs/user/atwang/data/spt_zhuang/cache/test"
}

sg_params = [global_params, train_params, gnn_params, loader_params, zhuang_params, saint_params]

sg2 = Dispatcher("sg2", sg_params, loaders.ZhuangBasic, models.SupRCGN, trainers.SupTrainer)
Dispatcher.variant(sg2, "sgt", [test_params])

sgc2 = Dispatcher("sgc2", sg_params, loaders.ZhuangBasicCellF, models.SupRCGN, trainers.SupTrainer)
Dispatcher.variant(sgc2, "sgct", [test_params])

sb2 = Dispatcher("sb2", sg_params, loaders.ZhuangBasicCellF, models.SupMLP, trainers.SupTrainer)
Dispatcher.variant(sb2, "sbt", [test_params])

lt_params = sg_params + [{"saint_num_steps": {"train": 500, "val": 1}}]
sg2 = Dispatcher("lt", lt_params, loaders.ZhuangBasicTest, models.SupRCGN, trainers.SupTrainer)


SaintRWTestLoader
