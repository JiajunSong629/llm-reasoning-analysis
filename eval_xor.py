import json
import random
from collections import defaultdict
from probes import LRProbe
from utils import DataManager


def calc_acc(datasets, labels, model_size, layer):
    accs = {}
    for label in labels:
        dm = DataManager()
        for dataset in datasets:
            dm.add_dataset(
                dataset_name=dataset,
                model_size=model_size,
                layer=layer,
                label=label,
                center=False,
                split=0.8,
            )
        acts, ys = dm.get("train")
        probe = LRProbe.from_data(acts, ys, bias=True)
        acts, ys = dm.get("val")
        acc = (probe(acts).round() == ys).float().mean().item()
        accs[label] = acc

    return accs


def test_alice():
    acc_all = defaultdict(dict)
    models = ["llama-3-8b", "llama-2-7b"]
    layers = [8, 10, 12, 14]
    datasets = ["cities_alice", "neg_cities_alice"]
    labels = [
        "has_alice",
        "has_not",
        "label",
        "has_alice xor has_not",
        "has_alice xor label",
        "has_not xor label",
        "has_alice xor has_not xor label",
    ]

    for model in models:
        for layer in layers:
            print(f"Calculating for {model} layer {layer}")
            acc = calc_acc(datasets, labels, model, layer)
            acc_all[model][layer] = acc

    with open("xor_acc.json", "w") as f:
        json.dump(acc_all, f)


def test_xor_letters():
    import pandas as pd

    model = "llama-3-8b"
    layers = [8, 10, 12, 14]
    datasets = ["xor_letters"]
    labels = pd.read_csv("datasets/xor_letters.csv").columns[1:].tolist()

    acc_all = defaultdict(dict)
    for layer in layers:
        print(f"Calculating for {model} layer {layer}")
        acc = calc_acc(datasets, labels, model, layer)
        acc_all[model][layer] = acc

    with open("xor_letters_acc.json", "w") as f:
        json.dump(acc_all, f)


if __name__ == "__main__":
    test_alice()
    test_xor_letters()
