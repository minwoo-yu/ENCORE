import copy
import os
import data
import torch
import yaml
import utils
import numpy as np
from model import common
import pandas as pd
from tqdm import tqdm
from AUHOC import AUHOC

def metrics_summ(metrics):
    psnr = metrics[0]
    ssim = metrics[1]
    auhoc = metrics[2]
    psnr_mean = round((np.mean(psnr).item()), 2)
    psnr_std = round(np.std(psnr).item(), 2)
    ssim_mean = round((np.mean(ssim).item()), 4)
    ssim_std = round(np.std(ssim).item(), 4)
    auhoc_mean = round((np.mean(auhoc).item()), 4)
    auhoc_std = round(np.std(auhoc).item(), 4)
    results = pd.DataFrame(
        {
            "PSNR": [psnr_mean, psnr_std],
            "SSIM": [ssim_mean, ssim_std],
            "AUHOC": [auhoc_mean, auhoc_std],
        }
    )
    return results


with open(os.path.join("yaml/test.yaml"), "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)


config["model"]["name"], config["model"]["level"] = "UNet", "B"
if config["model"]["name"] == "UNet":
    from model.unet import make_model
elif config["model"]["name"] == "DnCNN":
    from model.dncnn import make_model
elif config["model"]["name"] == "NAFNet":
    from model.nafnet import make_model
elif config["model"]["name"] == "Uformer":
    from model.uformer import make_model

base = f"{config['model']['name'].lower()}-{config['model']['level']}"

configs_list = [
    {"name": "Vanilla", "N_cov": False, "N_ch": 0, "backbone": "conv"},
    {"name": "+NADD", "N_cov": False, "N_ch": 1, "backbone": "conv"},
    {"name": "+COV", "N_cov": True, "N_ch": 1, "patch_size": 5, "kernel_size": 5, "backbone": "conv"},
    {"name": "+ENCORE", "N_cov": True, "N_ch": 1, "patch_size": 5, "kernel_size": 5, "backbone": "core"},
]

load_dirs = [
    base,
    base + "+NADD1",
    base + "+COV1_k5",
    base + "+OURS1_g2_k5",
]

config["model"]["pooling"] = True
config["model"]["num_blocks"], config["model"]["groups"], config["model"]["scale"] = [2, 2, 2, 2, 2], [2, 2, 2, 2, 2], 4

models = []
model_names = []
for i, cfg_item in enumerate(configs_list):
    cfg = copy.deepcopy(config)
    cfg["N_augment"].update({"N_cov": cfg_item["N_cov"], "N_ch": cfg_item["N_ch"]})
    cfg["model"]["backbone"] = cfg_item["backbone"]
    if cfg_item["N_cov"]:
        cfg["N_augment"].update({"patch_size": cfg_item["patch_size"], "kernel_size": cfg_item["kernel_size"]})

    model = make_model(cfg).cuda()
    
    load_dir = load_dirs[i]
    model_names.append(load_dir)
    state_dict = torch.load(os.path.join("experiment", load_dir, "model/model_best.pt"), map_location=torch.device(0))
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    models.append(model)

config["dataset"]["dose_level"], config["dataset"]["generation"] = 0.25, "cornish"
config["dataset"]["sigma_e"] = 2

config["dataset"]["test_data_path"] = "testdata/Mayo2016"
config["dataset"]["test_patient"] = ["L067", "L310"]
config["dataset"]["geometry"] = {
    "num_view": 1024,
    "num_det": 736,
    "det_interval": 1.2858,
    "sod": 595,
    "sdd": 1085.6,
}

# config["dataset"]["test_data_path"] = "testdata/Mayo2020"
# config["dataset"]["test_patient"] = ["L036", "L111", "L185", "L188", "L213", "L234"]
# config["dataset"]["geometry"] = {
#     "num_view": 1024,
#     "num_det": 888,
#     "det_interval": 1.0239,
#     "sod": 538.52,
#     "sdd": 946.746,
# }

reconstruction = common.NoiseRecon(config)
loader = data.Data(config, test_only=True)

all_names = ["Noisy"] + model_names
psnrs = [np.zeros(len(loader.loader_test)) for _ in range(len(all_names))]
ssims = [np.zeros(len(loader.loader_test)) for _ in range(len(all_names))]
auhocs = [np.zeros(len(loader.loader_test)) for _ in range(len(all_names))]
calc_sfrc = AUHOC(patch_size=64, air_thresh=-750, device="cuda")
with torch.no_grad():
    for i, (noisy, clean, metadata) in enumerate(tqdm(loader.loader_test)):
        noisy, clean = noisy.cuda(), clean.cuda()
        input, noise, recon = reconstruction(metadata, clean, noisy, N_ch=1)

        input = utils.normalize(input)
        noise = utils.normalize(noise, slope=100, intercept=0)

        input_denorm = utils.denormalize(input)
        outputs = [input_denorm] + [utils.denormalize(model(input.clone(), noise.clone())) for model in models]
        for idx, out in enumerate(outputs):
            psnrs[idx][i : i + len(recon)] = utils.cal_psnr(out, recon).item()
            ssims[idx][i : i + len(recon)] = utils.cal_ssim(out, recon).item()
            auhocs[idx][i : i + len(recon)] = calc_sfrc(out, recon).item()

results_dict = {}
for i in range(len(all_names)):
    results_i = metrics_summ((psnrs[i], ssims[i], auhocs[i]))
    results_dict[all_names[i]] = results_i
results = pd.concat(results_dict, names=["Methods", "Index"])

print(results)
