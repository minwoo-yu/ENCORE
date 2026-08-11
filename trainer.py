import os
import utils
import torch.nn.functional as F
import torch.nn as nn
import torch
from tqdm import tqdm
import time
import numpy as np
from torch.optim.lr_scheduler import MultiStepLR
from model import common
from importlib import import_module


class Trainer:
    def __init__(self, config, loader, ckp):
        self.config = config
        self.ckp = ckp
        self.device = torch.device("cpu" if config["cpu"] else "cuda")
        self.loader_train = (
            loader.loader_train
        )  # loader: Dataloader object -> has srdata object with self.svct and self.fvct
        self.loader_val = loader.loader_val
        self.loss = nn.MSELoss()

        module_model = import_module("model." + self.config["model"]["name"].lower())
        makemodel = getattr(module_model, "make_model")
        self.model = makemodel(config).to(self.device)

        if self.config["load"] != "":
            state_dict = torch.load(
                os.path.join("experiment", config["load"], "model/model_best.pt"), map_location=torch.device(0)
            )
            self.model.load_state_dict(state_dict, strict=True)
        # self.model = torch.compile(self.model)
        self.optimizer = utils.get_optimizer(
            config["optimizer"]["name"], self.model.parameters(), config["optimizer"]["lr"]
        )

        self.scheduler = MultiStepLR(
            self.optimizer, milestones=config["optimizer"]["milestones"], gamma=config["optimizer"]["gamma"]
        )  # decrease by half
        self.best_val_psnr = 0
        self.noise_n_recon = common.NoiseRecon(config)
        self.patch_coord = utils.make_coord(
            [self.config["dataset"]["patch_size"], self.config["dataset"]["patch_size"]], self.device
        )
        self.patch_coord = self.patch_coord.tile(self.config["dataset"]["batch_size"], 1, 1, 1)
        # Initialize lists to store loss values
        self.train_losses = []
        self.val_losses = []
        self.u_water = 0.0192867

    def train(self):
        epoch = self.scheduler.last_epoch
        train_loss = utils.Averager()
        t = utils.Timer()
        self.model.train()
        self.noise_n_recon.train()
        t.s()  # start timer
        for noisy, pos, metadata in tqdm(self.loader_train):
            noisy, pos = self.prepare(noisy, pos)

            input, noise, target = self.noise_n_recon(metadata, noisy, N_ch=self.config["N_augment"]["N_ch"])

            noise = utils.normalize(noise, slope=100, intercept=0) if noise is not None else None
            input = utils.normalize(input)
            target = utils.normalize(target)

            input, noise, target = common.get_recon_patch(input, target, self.patch_coord.clone(), pos, noise=noise)

            self.optimizer.zero_grad()
            output = self.model(input, noise)

            loss = self.loss(output, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 3)
            self.optimizer.step()
            train_loss.add(loss.item())

        self.train_losses.append(train_loss.item())
        train_time = t.t()  # end timer
        self.scheduler.step()
        # validation step
        if (epoch + 1) % self.config["test_every"] == 0:
            self.model.eval()
            self.noise_n_recon.eval()
            val_psnr = utils.Averager()
            val_ssim = utils.Averager()

            t.s()  # start timer
            with torch.no_grad():
                for i, (clean, metadata) in enumerate(tqdm(self.loader_val)):
                    clean = self.prepare(clean)[0]
                    e_noise = torch.randn_like(clean) * self.config["dataset"]["sigma_e"]
                    noisy = (torch.poisson(clean * self.config["dataset"]["dose_level"]) + e_noise).clamp_(min=1)
                    input, noise, target = self.noise_n_recon(
                        metadata, clean, ld_proj=noisy, N_ch=self.config["N_augment"]["N_ch"]
                    )
                    input = utils.normalize(input)
                    noise = utils.normalize(noise, slope=100, intercept=0) if noise is not None else None

                    output = self.model(input, noise)
                    output = utils.denormalize(output)
                    val_psnr.add(utils.cal_psnr(output, target).mean().item(), n=output.shape[0])
                    val_ssim.add(utils.cal_ssim(output, target).mean().item(), n=output.shape[0])

            avg_val_psnr = val_psnr.item()
            avg_val_ssim = val_ssim.item()
            self.val_losses.append(avg_val_psnr)
            val_time = t.t()  # end timer
            print("val_psnr is ... {}".format(avg_val_psnr))

            self.ckp.add_val_log(torch.tensor([avg_val_psnr]))

            if avg_val_psnr > self.best_val_psnr:
                self.best_val_psnr = avg_val_psnr
                self.ckp.save(self.model, is_best=True)  # save best model
            else:
                self.ckp.save(self.model, is_best=False)

    def prepare(self, *args):
        def _prepare(tensor):
            return tensor.to(self.device)

        return [_prepare(a) for a in args]

    def terminate(self):
        epoch = self.scheduler.last_epoch
        return epoch >= self.config["epochs"]
