import pydicom
import torch
import os
import numpy as np
import yaml
from tqdm import tqdm
import leapctype

device = torch.device("cuda")
view = 1024

patient_ID = ["L067", "L096", "L310", "L143", "L291", "L109", "L192", "L506"]  # Mayo2016


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
    with open(os.path.join("yaml/generation" + ".yaml"), "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config["dataset"]["geometry"] = {
        "num_view": 1024,
        "num_det": 736,
        "det_interval": 1.2858,
        "sod": 595,
        "sdd": 1085.6,
    }
    N_in = float(config["dataset"]["N_in"])
    patient_ID = config["dataset"]["train_patient"] + config["dataset"]["val_patient"]
    for id in patient_ID:
        root_dir = os.path.join("/Mayo2016", id)
        clean_path = os.path.join("traindata", id, "clean")
        noisy_path = os.path.join("traindata", id, "noisy")
        os.makedirs(os.path.join(clean_path), exist_ok=True)
        os.makedirs(os.path.join(noisy_path), exist_ok=True)
        print("save path is {}".format(clean_path))
        projector = leapctype.tomographicModels()
        for root, _, filenames in os.walk(root_dir):
            if "full_1mm" in root:
                if id == "L096":
                    filenames = filenames[70:]
                for file_name in tqdm(filenames):
                    data = pydicom.dcmread(os.path.join(root, file_name))
                    img_npy = data.pixel_array * data.RescaleSlope
                    img = torch.FloatTensor(img_npy.copy()).unsqueeze(0).to(device) - 24
                    offset = np.array(data.ReconstructionTargetCenterPatient) - np.array(
                        data.DataCollectionCenterPatient
                    )
                    x = y = torch.linspace(-255.5, 255.5, 512, device="cuda") * data.PixelSpacing[0]
                    grid_x, grid_y = torch.meshgrid(x, y, indexing="xy")
                    fov = torch.sqrt((grid_x + offset[0]) ** 2 + (grid_y + offset[1]) ** 2)
                    fov_valid = fov >= (data.DataCollectionDiameter / 2)
                    img[fov_valid.unsqueeze(0)] = 0

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
                    save_clean = {
                        "projection_data": clean_proj.squeeze().cpu().numpy().astype(np.uint32),
                        "metadata": metadata,
                    }
                    np.save(os.path.join(clean_path, file_name[:-4] + ".npy"), save_clean)

                    e_noise = torch.randn_like(clean_proj) * config["dataset"]["sigma_e"]
                    nd_proj = (torch.poisson(clean_proj) + e_noise).clamp_(min=1)
                    save_nd = {
                        "projection_data": nd_proj.squeeze().cpu().numpy().astype(np.uint32),
                        "metadata": metadata,
                    }
                    np.save(os.path.join(noisy_path, file_name[:-4] + ".npy"), save_nd)

        print("finish {}, KVP: {}".format(id, int(data.KVP)))


if __name__ == "__main__":
    main()
