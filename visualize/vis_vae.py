from model.models import SupCVAE
from model.loaders import Synth3Layer

def load_model(model_cls, params, model_state):
	m = model_cls(**params)
	m.load_state_dict(model_state)
	return m

def sample_model(loader, vae_model, device):
	for data in loader.test_sampler:
		data.to(device)
		num_samples = data.x.shape[0]
		out = model.sample_coords(data)
		out_coords = out["coords"]

		cell_mask = data.cell_mask.detach().cpu().numpy()
		exp = data.x.detach().cpu().numpy()
		coords_true = data.pos.detach().cpu().numpy()
		coords_pred = out_coords.detach().cpu().numpy()
		num_samples = exp.shape[0]
		for ind in range(num_samples):
			print(exp[i]) ####


def vis_vae(loader, vae_model, dname):
	device = dname if dname == "cpu" else f"cuda:{dname}" 
	sample_model(loader, vae_model, device)

if __name__ == '__main__':
    data_dir = "/dfs/user/atwang/data/analyses/st_gnn"
    result_dir = "/dfs/user/atwang/results/st_gnn_results/spt_zhuang/sup/training"

    loader = Synth3Layer
    vae_model = SupCVAE

    dname = sys.argv[1]

    vis_vae(loader, vae_model, dname)
