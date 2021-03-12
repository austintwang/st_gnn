import os
import numpy as np
import model.models as models
import model.loaders as loaders
import model.trainers as trainers

class Dispatcher(object):
    names = {}
    def __init__(self, name, paramlist, model_cpnts, loader_cls, model_cls, trainer_cls):
        Dispatcher.names[name] = self
        self.name = name

        self.params = {}
        for i in paramlist:
            self.params.update(i)

        self.model_cpnts = model_cpnts

        self.loader_cls = loader_cls
        self.model_cls = model_cls
        self.trainer_cls = trainer_cls

    @classmethod
    def variant(cls, other, name, paramlist):
        paramlist = [other.params] + paramlist
        return cls(name, paramlist, other.model_cpnts, other.loader_cls, other.model_cls, other.trainer_cls)

    def dispatch(self, device, clear_cache):
        self.params.update({"name": self.name, "device": device, "clear_cache": clear_cache})

        loader = self.loader_cls(**self.params)
        model = self.model_cls(loader.node_in_channels, self.model_cpnts, **self.params)
        trainer = self.trainer_cls(model, loader, **self.params)
        trainer.run()


global_params = {
    "num_workers": 8,
}
        
train_params = {
    "num_epochs": 1000,
    "learning_rate": 1e-3,
    "early_stop_min_delta": 0.001,
    "early_stop_hist_len": 10,
    "dropout_prop": 0.1,
    "struct_layers_out_chnls": [64],
    "min_dist": 1e-4,
    "grad_clip_norm": 1.,
    "results_dir": "/dfs/user/atwang/data/analyses/st_gnn"
}

gnn_params = {
    "emb_layers_out_chnls": [256, 256]
}

loader_params = {
    "batch_size": 400,
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
    "saint_num_steps": {"train": 625, "val": 150},
    "saint_sample_coverage": 100,
    "loader_cache_dir": "/dfs/user/atwang/data/spt_zhuang/cache/saint"
}

test_params = {
    "saint_sample_coverage": 2, 
    "saint_num_steps": {"train": 10, "val": 10, "test": 10},
    "loader_cache_dir": "/dfs/user/atwang/data/spt_zhuang/cache/test",
    "debug": True
}

sg_components = {
    "emb": models.EmbMLP,
    "struct": models.StructCoords,
}

sg_params = [global_params, train_params, gnn_params, loader_params, zhuang_params, saint_params]

# sg2 = Dispatcher("sg2", sg_params, {"emb": models.EmbRGCN, "struct": models.StructPairwiseLN}, loaders.ZhuangBasic, models.SupRCGN, trainers.SupTrainer)
# Dispatcher.variant(sg2, "sgt", [test_params])

# sgc2 = Dispatcher("sgc2", sg_params, loaders.ZhuangBasicCellF, models.SupRCGN, trainers.SupTrainer)
# Dispatcher.variant(sgc2, "sgct", [test_params])

sb2_components = {
    "emb": models.EmbMLP,
    "struct": models.StructCoords,
}

# sb2 = Dispatcher("sb2", sg_params, sb2_components, loaders.ZhuangBasicCellF, models.SupFF, trainers.SupTrainer)
# Dispatcher.variant(sb2, "sbt", [test_params])


# sg2s = Dispatcher("sg2s", sg_params, loaders.ZhuangBasic, models.SupSRCGN, trainers.SupMSETrainer)
# Dispatcher.variant(sg2s, "sgts", [test_params])

# sgc2s = Dispatcher("sgc2s", sg_params, loaders.ZhuangBasicCellF, models.SupSRCGN, trainers.SupMSETrainer)
# Dispatcher.variant(sgc2s, "sgcts", [test_params])

sb2s_components = {
    "emb": models.EmbMLP,
    "struct": models.StructPairwiseN,
}

sb2s = Dispatcher("sb2s", sg_params, sb2s_components, loaders.ZhuangBasicCellF, models.SupFF, trainers.SupMSETrainer)
Dispatcher.variant(sb2s, "sbts", [test_params])


# sglr_params = sg_params + [{"grad_clip_norm": 0.0001, "learning_rate": 1e-100,}]

# sg2lr = Dispatcher("sg2lr", sglr_params, loaders.ZhuangBasic, models.SupLRRCGN, trainers.SupMSETrainer)
# Dispatcher.variant(sg2lr, "sgtlr", [test_params])

# sgc2lr = Dispatcher("sgc2lr", sglr_params, loaders.ZhuangBasicCellF, models.SupLRRCGN, trainers.SupMSETrainer)
# Dispatcher.variant(sgc2lr, "sgctlr", [test_params])

# sb2lr = Dispatcher("sb2lr", sglr_params, loaders.ZhuangBasicCellF, models.SupLRMLP, trainers.SupMSETrainer)
# Dispatcher.variant(sb2lr, "sbtlr", [test_params])


# sgbin50_params = sg_params + [{"adj_thresh": 50}]

# sg2bin50 = Dispatcher("sg2bin50", sgbin50_params, loaders.ZhuangBasic, models.SupBinRCGN, trainers.SupBinTrainer)
# Dispatcher.variant(sg2bin50, "sgtbin50", [test_params])

# sgc2bin50 = Dispatcher("sgc2bin50", sgbin50_params, loaders.ZhuangBasicCellF, models.SupBinRCGN, trainers.SupBinTrainer)
# Dispatcher.variant(sgc2bin50, "sgctbin50", [test_params])

# sb2bin50 = Dispatcher("sb2bin50", sgbin50_params, loaders.ZhuangBasicCellF, models.SupBinMLP, trainers.SupBinTrainer)
# Dispatcher.variant(sb2bin50, "sbtbin50", [test_params])


# sgbin100_params = sg_params + [{"adj_thresh": 100}]

# sg2bin100 = Dispatcher("sg2bin100", sgbin100_params, loaders.ZhuangBasic, models.SupBinRCGN, trainers.SupBinTrainer)
# Dispatcher.variant(sg2bin100, "sgtbin100", [test_params])

# sgc2bin100 = Dispatcher("sgc2bin100", sgbin100_params, loaders.ZhuangBasicCellF, models.SupBinRCGN, trainers.SupBinTrainer)
# Dispatcher.variant(sgc2bin100, "sgctbin100", [test_params])

# sb2bin100 = Dispatcher("sb2bin100", sgbin100_params, loaders.ZhuangBasicCellF, models.SupBinMLP, trainers.SupBinTrainer)
# Dispatcher.variant(sb2bin100, "sbtbin100", [test_params])


# sgbin500_params = sg_params + [{"adj_thresh": 500}]

# sg2bin500 = Dispatcher("sg2bin500", sgbin500_params, loaders.ZhuangBasic, models.SupBinRCGN, trainers.SupBinTrainer)
# Dispatcher.variant(sg2bin500, "sgtbin500", [test_params])

# sgc2bin500 = Dispatcher("sgc2bin500", sgbin500_params, loaders.ZhuangBasicCellF, models.SupBinRCGN, trainers.SupBinTrainer)
# Dispatcher.variant(sgc2bin500, "sgctbin500", [test_params])

# sb2bin500 = Dispatcher("sb2bin500", sgbin500_params, loaders.ZhuangBasicCellF, models.SupBinMLP, trainers.SupBinTrainer)
# Dispatcher.variant(sb2bin500, "sbtbin500", [test_params])

# lt_params = sg_params + [test_params, {"saint_num_steps": {"train": 5, "val": 1}, "saint_sample_coverage": 500,}]
# sg2 = Dispatcher("lt", lt_params, loaders.ZhuangBasicTest, models.SupRCGN, trainers.SupTrainer)


vae_train_params = {
    "num_epochs": 500,
    "learning_rate": 1e-3,
    "early_stop_min_delta": 0.001,
    "early_stop_hist_len": 10,
    "dropout_prop": 0.1,
    "grad_clip_norm": 1.,
    # "vae_struct_nll_std": 10,
    # "vae_exp_nll_std": 1.,
    # "vae_sup_nll_std": 1e4,
    "vae_struct_nll_w": 10.,
    "vae_exp_nll_w": 1.,
    "vae_sup_nll_w": 1e-5,
    "vae_struct_kl_w": 1.,
    "vae_exp_kl_w": 0.1,
    "results_dir": "/dfs/user/atwang/data/analyses/st_gnn"
}

synth_vae_train_params = {
    "early_stop_min_delta": -np.inf,
    "vae_struct_nll_w": 100.,
    "vae_exp_nll_w": 1.,
    "vae_sup_nll_w": 0.,
    "vae_struct_kl_w": 0.1,
    "vae_exp_kl_w": 0.001,
}

synth_vae_model_params = {
    "emb_layers_out_chnls": [16, 16],
    "struct_layers_out_chnls": [16, 16],
    "aux_struct_enc_layers_out_chnls": [16, 16],
    "aux_exp_dec_layers_out_chnls": [16, 16],
    "vae_enc_add_chnls": [32],
    "vae_latent_dim": 3,
}

synth_loader_params = {
    "batch_size": 50,
    "train_prop": 0.8,
}

synth_params = {
    "synth_num_points": 10000
}

synth_saint_params = {
    "saint_num_steps": {"train": 200, "val": 40, "test": 200},
}

noexp_params = {"vae_exp_nll_w": 0}

vae_params = [global_params, vae_train_params, synth_vae_train_params, synth_vae_model_params, synth_loader_params, synth_params, saint_params, synth_saint_params]

vae_components = {
    "emb": models.EmbMLP,
    "struct": models.StructCoords,
    "struct_enc": models.AuxStructEncMLP,
    "exp_dec": models.AuxExpDecMLP,
}

vs = Dispatcher("vs", vae_params, vae_components, loaders.Synth3Layer, models.SupCVAE, trainers.CVAETrainer)
Dispatcher.variant(vs, "vst", [test_params])
Dispatcher.variant(vs, "vsae", [noexp_params])

vsas = Dispatcher("vsas", vae_params, vae_components, loaders.Synth3Layer, models.SupCVAENS, trainers.CVAETrainer)
Dispatcher.variant(vsas, "vsaes", [noexp_params])

vae_model_params = {
    "emb_layers_out_chnls": [256, 256],
    "struct_layers_out_chnls": [256, 256],
    "aux_struct_enc_layers_out_chnls": [256, 256],
    "aux_exp_dec_layers_out_chnls": [256, 256],
    "vae_enc_add_chnls": [128],
    "vae_latent_dim": 32,
    "vae_lvar_scale": 0.01,
}

vae_params = [global_params, vae_train_params, vae_model_params, loader_params, zhuang_params, saint_params]

vb2 = Dispatcher("vb2", vae_params, vae_components, loaders.ZhuangBasicCellF, models.SupCVAE, trainers.CVAETrainer)
Dispatcher.variant(vb2, "vbt", [test_params])
