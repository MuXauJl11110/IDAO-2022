import json
import os
import shutil
from doctest import UnexpectedException
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from pymatgen.core import Structure
from pymatgen.io import cif
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

from cgcnn.cgcnn.data import CIFData, collate_pool
from cgcnn.cgcnn.model import CrystalGraphConvNet


class Normalizer(object):
    """Normalize a Tensor and restore it later."""

    def __init__(self, tensor):
        """tensor is taken as a sample to calculate the mean and std"""
        self.mean = torch.mean(tensor)
        self.std = torch.std(tensor)

    def norm(self, tensor):
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor):
        return normed_tensor * self.std + self.mean

    def state_dict(self):
        return {"mean": self.mean, "std": self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict["mean"]
        self.std = state_dict["std"]


def read_pymatgen_dict(file):
    with open(file, "r") as f:
        d = json.load(f)
    return Structure.from_dict(d)


def load_data(data_path: str, test: bool):
    def load_single_data(path):
        embd, ids, target = sorted(os.listdir(path))
        embeddings = np.load(path + "/" + embd)
        ids = np.load(path + "/" + ids)
        targets = np.load(path + "/" + target)
        return embeddings, ids, targets

    X = np.array([])
    if not test:
        y = np.array([])
    ids = np.array([])
    for batch in tqdm(sorted(os.listdir(data_path)), desc="Loading data"):
        embeddings, ids_, targets = load_single_data(data_path + "/" + batch)
        targets = targets.reshape(targets.shape[0])
        X = np.vstack([embeddings, X]) if X.size else embeddings
        if not test:
            y = np.concatenate((targets, y))
        ids = np.concatenate((ids_, ids))

    if test:
        return X, ids
    else:
        return X, y, ids


def prepare_data(src_path: str, dist_path: str, json_path: str, target_file: str, test: bool):
    src_path = Path(src_path)
    if not src_path.exists():
        raise ValueError("Specified path doesn't exist!")

    dist_path = Path(dist_path)
    if dist_path.exists():
        if not dist_path.is_dir():
            dist_path.unlink()
            Path(dist_path).mkdir(parents=True, exist_ok=True)
    else:
        Path(dist_path).mkdir(parents=True, exist_ok=True)

    if not test:
        targets = pd.read_csv(src_path / "targets.csv")
        targets.to_csv(dist_path / target_file, index=False, header=False)

    for item in tqdm((src_path / "structures").iterdir(), desc="writing to .cif files"):
        struct = read_pymatgen_dict(item)
        cif_file = cif.CifWriter(struct)
        cif_file.write_file(dist_path / item.with_suffix(".cif").name)

    if test:
        ids = []
        values = []
        for item in tqdm(dist_path.iterdir(), desc="generating test data"):
            ids.append(item.stem)
            values.append(0.0)
        data = pd.DataFrame(data={"_id": ids, "value": values})
        data.to_csv(dist_path / target_file, index=False, header=False)

    json_path = Path(json_path)
    shutil.copy(json_path, dist_path / json_path.name)


def validate(val_loader, model, cuda: bool, name: str):
    model.eval()
    for i, (input, target, batch_cif_ids) in enumerate(tqdm(val_loader, desc=f"validation {name}")):
        with torch.no_grad():
            if cuda:
                input_var = (
                    Variable(input[0].cuda(non_blocking=True)),
                    Variable(input[1].cuda(non_blocking=True)),
                    input[2].cuda(non_blocking=True),
                    [crys_idx.cuda(non_blocking=True) for crys_idx in input[3]],
                )
            else:
                input_var = (Variable(input[0]), Variable(input[1]), input[2], input[3])
        # compute output
        output = model(*input_var)
        np_output = output.detach().cpu().numpy()
        target = target.numpy()
        batch_cif_ids = np.array(batch_cif_ids)
        data_path = name + "/batch_" + f"{i:06d}"
        os.makedirs(data_path)
        np.save(data_path + "/embedding.npy", np_output)
        np.save(data_path + "/target.npy", target)
        np.save(data_path + "/ids.npy", batch_cif_ids)


def prepare_model(model_args: dict, cuda: bool, dataset_path: str):
    class InterCrystalGraphConvNet(CrystalGraphConvNet):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)

        def _forward_inter(self, atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx):
            atom_fea = self.embedding(atom_fea)
            for conv_func in self.convs:
                atom_fea = conv_func(atom_fea, nbr_fea, nbr_fea_idx)
            crys_fea = self.pooling(atom_fea, crystal_atom_idx)
            return crys_fea

        def forward(self, atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx):
            return self._forward_inter(atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx)

    modelpath = Path(model_args["path"])
    if modelpath.exists():
        if modelpath.is_file():
            print(f"=> loading model params '{modelpath}'")
            # Load all tensors onto the CPU
            model_checkpoint = torch.load(modelpath, map_location=torch.device("cpu"))
            args = model_checkpoint["args"]
            print(f"=> loaded model params '{modelpath}'")

            # load data
            dataset = CIFData(dataset_path)
            collate_fn = collate_pool
            data_loader = DataLoader(
                dataset,
                batch_size=model_args["batch_size"],
                shuffle=True,
                num_workers=model_args["num_workers"],
                collate_fn=collate_fn,
                pin_memory=cuda,
            )
            # build model
            structures, _, _ = dataset[0]
            orig_atom_fea_len = structures[0].shape[-1]
            nbr_fea_len = structures[1].shape[-1]
            model = InterCrystalGraphConvNet(
                orig_atom_fea_len=orig_atom_fea_len,
                nbr_fea_len=nbr_fea_len,
                atom_fea_len=args["atom_fea_len"],
                n_conv=args["n_conv"],
                h_fea_len=args["h_fea_len"],
                n_h=args["n_h"],
                classification=(True if args["task"] == "classification" else False),
            )
            if cuda:
                model.cuda()

            return model, data_loader
        else:
            print(f"=> {modelpath.name} isn't a file")
    else:
        print(f"=> no model params found at '{modelpath}'")
    return None, None


def make_submission(res, frame, name):
    f1 = frame.assign(predictions=res)
    f1.to_csv(name)

    return f1


def main(config):
    if config["data_preprocessing"]["train"]:
        print(f"=> preparing data from {config['datapath']} to {config['data_preprocessing']['new_datapath']}")
        prepare_data(
            config["datapath"],
            config["data_preprocessing"]["new_datapath"],
            config["data_preprocessing"]["json_path"],
            config["data_preprocessing"]["target_file"],
            test=False,
        )
    if config["data_preprocessing"]["test"]:
        print(
            f"=> preparing test data from {config['test_datapath']} to {config['data_preprocessing']['new_test_datapath']}"
        )
        prepare_data(
            config["test_datapath"],
            config["data_preprocessing"]["new_test_datapath"],
            config["data_preprocessing"]["json_path"],
            config["data_preprocessing"]["target_file"],
            test=True,
        )
    cuda = not config["model"]["disable_cuda"] and torch.cuda.is_available()
    print(f"=> cuda: {cuda}")

    if config["embeddings"]["train"]:
        print("=> start preparing train embeddings\n")
        model, loader = prepare_model(
            model_args=config["model"], cuda=cuda, dataset_path=config["data_preprocessing"]["new_datapath"]
        )
        if model is not None:
            validate(loader, model, cuda, config["embeddings"]["train_name"])
        else:
            raise UnexpectedException("Oopsie! Something went wrong. Model hasn't been prepared.")
        print("\n=> end preparing train embeddings")
    if config["embeddings"]["test"]:
        print("=> start preparing test embeddings\n")
        model, loader = prepare_model(
            model_args=config["model"], cuda=cuda, dataset_path=config["data_preprocessing"]["new_test_datapath"]
        )
        if model is not None:
            validate(loader, model, cuda, config["embeddings"]["test_name"])
        else:
            raise UnexpectedException("Oopsie! Something went wrong. Model hasn't been prepared.")
        print("\n=> end preparing test embeddings")


if __name__ == "__main__":
    with open("config.yaml") as file:
        config = yaml.safe_load(file)
    main(config)
