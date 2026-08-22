import pydicom
import torch
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
import numpy as np
import time
import yaml
from tqdm import tqdm
import leapctype

device = torch.device("cuda")
view = 1024


def geometry_setting(leap, metadata):
    leap.set_fanbeam(
        int(metadata[5].item()),  ## num_view
        1,
        int(metadata[6].item()),  ## num_det
        1,
        metadata[4].item(),  ## det_interval
        0,
        0.5 * (metadata[6].item() - 1),
        np.linspace(0, 360, int(metadata[5].item()), endpoint=False, dtype="float32"),
        metadata[1].item(),  ## distance source to patient
        metadata[2].item(),  ## distance source to detector
    )
    leap.set_volume(
        512,
        512,
        1,
        voxelWidth=metadata[3],  ## recon_interval
        voxelHeight=1,
    )
    return leap

def main():
    with open(os.path.join("/root/code/2D/yaml/generation" + ".yaml"), "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config["dataset"]["dose_level"], config["dataset"]["sigma_e"] = 0.1, 2
    config["dataset"]["geometry"] = {
        "data_type": "DICOM",
        "num_view": 1024,
        "num_det": 888,
        "det_interval": 1.0239,
        "sod": 538.52,
        "sdd": 946.746,
    }
    N_in = float(config["dataset"]["N_in"])
    patient_ID = ["L036", "L111", "L185", "L188", "L213", "L234"]
    # patient_ID = ["L235"]
    print("dose_level is {}, sigma_e is {}".format(config["dataset"]["dose_level"], config["dataset"]["sigma_e"]))
    for id in patient_ID:
        save_path = os.path.join("testdata/Mayo2020", id)
        # os.makedirs(os.path.join(save_path, "noisy", str(config["dataset"]["dose_level"]) + "_blur"), exist_ok=True)
        os.makedirs(os.path.join(save_path, "noisy", str(config["dataset"]["dose_level"])), exist_ok=True)
        os.makedirs(os.path.join(save_path, "clean"), exist_ok=True)
        print("save path is {}".format(save_path))
        projector = leapctype.tomographicModels()
        folder = sorted(os.listdir(os.path.join("/Mayo2020", id)))[-1]
        imgfolder = sorted(os.listdir(os.path.join("/Mayo2020", id, folder)))[-1]
        filenames = sorted(os.listdir(os.path.join("/Mayo2020", id, folder, imgfolder)))
        for file_name in tqdm(filenames):
            try:
                idx = int(file_name.split("-")[-1].split(".")[0])
            except (ValueError, IndexError):
                idx = -1

            if id == "L111" and 122 <= idx <= 132:
                continue
            if id == "L234" and 77 <= idx <= 109:
                continue
            if id == "L185" and 50 <= idx <= 135:
                continue

            data = pydicom.dcmread(os.path.join("/Mayo2020", id, folder, imgfolder, file_name))
            img_npy = data.pixel_array * data.RescaleSlope
            mask = img_npy == -2000
            img_npy[mask] = 24
            img = torch.FloatTensor(img_npy.copy()).unsqueeze(0).to(device) - 24
            metadata = np.array(
                [
                    data.KVP,
                    data.DistanceSourceToPatient,
                    data.DistanceSourceToDetector,
                    data.PixelSpacing[0],
                    config["dataset"]["geometry"]["det_interval"],
                    config["dataset"]["geometry"]["num_view"],
                    config["dataset"]["geometry"]["num_det"],
                ]
            )
            projector = geometry_setting(projector, metadata)

            clean_proj = torch.zeros(int(metadata[5]), img.shape[0], int(metadata[6]), device=img.device)
            u_water = 0.0192867 if int(data.KVP) == 120 else 0.0205888
            clean_proj = projector.project(clean_proj, img / 1000 * u_water)
            clean_proj = (torch.exp(-clean_proj) * N_in).clamp_(min=1)
            save_data = {"projection_data": clean_proj.squeeze().cpu().numpy().astype(np.uint32), "metadata": metadata}
            np.save(os.path.join(save_path, "clean", file_name[:-4] + ".npy"), save_data)

            e_noise = config["dataset"]["sigma_e"] * torch.randn_like(clean_proj)
            ld_proj = (torch.poisson(clean_proj * config["dataset"]["dose_level"]) + e_noise).clamp(min=1)
            save_data = {"projection_data": ld_proj.squeeze().cpu().numpy().astype(np.uint32), "metadata": metadata}
            np.save(
                os.path.join(save_path, "noisy", str(config["dataset"]["dose_level"]), file_name[:-4] + ".npy"),
                save_data,
            )
        print("finish {}, KVP: {}".format(id, int(metadata[0])))


if __name__ == "__main__":
    main()
