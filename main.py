import os
import torch
import argparse
import yaml
import utils
import data
from trainer import Trainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="/root/code/2D/yaml/unet+encore")
    parser.add_argument("--save", type=str, default="test")
    parser.add_argument("--save_dir", type=str, default="experiment")
    parser.add_argument("--load", type=str, default="")

    args = parser.parse_args()

    with open(os.path.join(args.config + ".yaml"), "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        print("config loaded, config_path: {}".format(os.path.join(args.config + ".yaml")))
    # config["dataset"]["anatomy"] = args.anatomy
    config["load"] = args.load
    torch.manual_seed(config["seed"])

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = utils.checkpoint(args.save_dir, args.save)

    if checkpoint.ok:
        print("Data loading Started!")
        loader = data.Data(config, test_only=False)  # If train and val. True when test
        print("Data loading Completed!")

        t = Trainer(config, loader, checkpoint)
        print("Training Started!")
        while not t.terminate():  # terminate when current epoch >= config['epochs']
            t.train()

        print("Training Ended!")


if __name__ == "__main__":
    main()
