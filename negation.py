import os
import torch
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForCausalLM

from data import get_seqs_happy_negation, get_seqs_math

MODELS = {
    "llama-3-8b": "/home/jiajun/.cache/huggingface/hub/models--meta-llama--Llama-3-8b-hf",
    "gemma-2-2b": "google/gemma-2-2b",
    "gemma-2-2b-it": "google/gemma-2-2b-it",
    "gemma-7b-it": "google/gemma-7b-it",
    "gemma-2-9b": "google/gemma-2-9b",
    "gpt2": "gpt2",
    "mistral-7b": "mistralai/Mistral-7B-v0.1",
    "mistral-7b-it": "mistralai/Mistral-7B-Instruct-v0.1",
}

SEQ_FUNS = {
    "happy_negation": get_seqs_happy_negation,
    "math": get_seqs_math,
}


def completion(model, tokenizer, seqs, template, out_dir, fig_dir):
    with open(f"{out_dir}/completion_template_{template}.txt", "w") as f:
        for seq in seqs:
            inputs = tokenizer(seq, return_tensors="pt", padding=True).to(model.device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs, max_new_tokens=10, pad_token_id=tokenizer.eos_token_id
                )
            for output in outputs:
                f.write(tokenizer.decode(output, skip_special_tokens=True) + "\n")
                f.write("-------------------------------------\n")

            f.write("\n\n")


def cosine_sim(model, tokenizer, seqs, template, out_dir, fig_dir):
    def calc_cos_sim(u, v):
        o = torch.dot(u, v) / (torch.linalg.norm(u) * torch.linalg.norm(v))
        return o.item()

    batch_size, n_comparison = len(hh), len(hh[0])

    layer = (
        20
        if model.config.num_hidden_layers > 20
        else int(0.8 * model.config.num_hidden_layers)
    )
    hh = []
    logits = []

    for seq in seqs:
        inputs = tokenizer(seq, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            out = model(**inputs)
        h = out.hidden_states[layer][:, -1].to("cpu").float()
        hh.append(h)
        logits.append(out.logits[:, -1].to("cpu").float())

    svals = torch.zeros(n_comparison - 1, batch_size)
    for k in range(n_comparison - 1):
        h_diff0 = torch.stack(
            [hh[j][k + 1] - hh[j][0] for j in range(len(hh))], dim=1
        )  # (d_model, batch_size)
        _, s, _ = torch.linalg.svd(h_diff0[:, :-1])
        svals[k] = s

    sims = torch.zeros(n_comparison - 1, batch_size)
    sims[0] = torch.tensor(
        [calc_cos_sim(hh[j][1] - hh[j][0], hh[j][2] - hh[j][0]) for j in range(len(hh))]
    )
    sims[1] = torch.tensor(
        [calc_cos_sim(hh[j][1] - hh[j][0], hh[j][3] - hh[j][0]) for j in range(len(hh))]
    )
    sims[2] = torch.tensor(
        [calc_cos_sim(hh[j][2] - hh[j][0], hh[j][3] - hh[j][0]) for j in range(len(hh))]
    )

    fig, axs = plt.subplots(1, 2, figsize=(13, 6))
    labels = ["Un-0", "Not-0", "UnNot-0"]
    for k in range(n_comparison - 1):
        axs[0].plot(svals[k], linewidth=2, label=labels[k])
    axs[0].set_title("Svals of embedding diff", weight="bold")
    axs[0].legend()
    sns.heatmap(sims.numpy(), ax=axs[1])
    axs[1].set_xlabel("Prompt idx", weight="bold")
    axs[1].set_ylabel("Pairs", weight="bold")
    axs[1].set_yticklabels(["Un-0/Not-0", "Un-0/UnNot-0", "Not-UnNot"])
    axs[1].set_title(f"Cos sim of embedding diff, Layer {layer}", weight="bold")
    plt.savefig(f"{fig_dir}/cosine_sim_template_{template}.png")
    plt.close()


def prediction_probs(model, tokenizer, seqs, template, out_dir, fig_dir):
    target_tokens_ids = tokenizer.convert_tokens_to_ids(["Yes", "No", "yes", "no"])

    batch_size, n_comparison = len(seqs), len(seqs[0])

    layer = (
        20
        if model.config.num_hidden_layers > 20
        else int(0.8 * model.config.num_hidden_layers)
    )
    hh = []
    logits = []

    for seq in seqs:
        inputs = tokenizer(seq, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            out = model(**inputs)
        h = out.hidden_states[layer][:, -1].to("cpu").float()
        hh.append(h)
        logits.append(out.logits[:, -1].to("cpu").float())

    logits_subset = torch.stack(logits, dim=2)[:, target_tokens_ids, :]
    probs = torch.softmax(logits_subset, dim=1)

    y = np.zeros((n_comparison, 5))
    y[:, 0] = np.arange(n_comparison)
    y[:, 1:] = probs[:, :, :-1].mean(dim=2).numpy()
    df = pd.DataFrame(y, columns=["prompt type", "Yes", "No", "yes", "no"])
    ax = df.plot(x="prompt type", y=["Yes", "No", "yes", "no"], kind="bar", rot=0)
    ax.set_title("Prediction Probs", weight="bold")
    # ax.set_xticklabels(seqs[0], weight="bold", rotation=45, ha='right')
    # ax.set_xticklabels(["0", "Un", "Not", "UnNot"], weight="bold")
    plt.savefig(f"{fig_dir}/prediction_probs_template_{template}.png")
    plt.close()


def diff_analysis(model, tokenizer, seqs, template, out_dir, fig_dir):
    n_pc = 5
    if model.config.model_type == "gpt2":
        norm = model.transformer.ln_f
        lm_head = model.lm_head
    else:
        norm = model.model.norm
        lm_head = model.lm_head

    batch_size, n_comparison = len(hh), len(hh[0])

    layer = (
        20
        if model.config.num_hidden_layers > 20
        else int(0.8 * model.config.num_hidden_layers)
    )
    hh = []
    logits = []

    for seq in seqs:
        inputs = tokenizer(seq, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            out = model(**inputs)
        h = out.hidden_states[layer][:, -1].to("cpu").float()
        hh.append(h)
        logits.append(out.logits[:, -1].to("cpu").float())

    with open(f"{out_dir}/analysis_template_{template}.txt", "w") as f:
        for k in range(3):
            h_diff0 = torch.stack(
                [hh[j][k + 1] - hh[j][0] for j in range(len(hh))], dim=1
            ).T

            h_diff0_normed = norm(h_diff0.to(model.device, dtype=torch.bfloat16))
            h_diff0_normed = h_diff0_normed - h_diff0_normed.mean(dim=0, keepdim=True)
            u_normed, s_normed, vt_normed = torch.linalg.svd(
                h_diff0_normed.cpu().float()
            )

            h_diff0 = h_diff0.to(model.device, dtype=torch.bfloat16)
            logits = lm_head(norm(h_diff0))

            for seq_index, logit in enumerate(logits):
                f.write(f"Sequence {seq_index} Top3: ")
                topk = torch.topk(logit, 3, dim=-1).indices
                f.write(" ".join([tokenizer.decode(t) for t in topk]) + "\n")
            f.write("--------------------------------------\n")

            for pc_index in range(n_pc):
                f.write(f"PC {pc_index} Top3: ")
                logits_pc = lm_head(
                    vt_normed[pc_index].to(model.device, dtype=torch.bfloat16)
                )
                topk = torch.topk(logits_pc, 3, dim=-1).indices
                f.write(" ".join([tokenizer.decode(t) for t in topk]) + "\n")
            f.write("#######################################\n\n")


def main(model_name, seq_fun: str, out_dir=None, fig_dir=None):
    model = AutoModelForCausalLM.from_pretrained(
        MODELS[model_name],
        output_hidden_states=True,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    tokenizer = AutoTokenizer.from_pretrained(MODELS[model_name])

    if out_dir is None:
        # replace - with _
        out_dir = f"out/{model_name.replace('-', '_')}"
    if fig_dir is None:
        fig_dir = f"{out_dir}/figs"

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    for template in range(4):
        seqs = SEQ_FUNS[seq_fun](template)

        ## test text generation
        completion(model, tokenizer, seqs, template, out_dir, fig_dir)

        # prediction probs
        prediction_probs(model, tokenizer, seqs, template, out_dir, fig_dir)

        if seq_fun == "happy_negation":
            # cosine sim
            cosine_sim(model, tokenizer, seqs, template, out_dir, fig_dir)

            # diff analysis
            diff_analysis(model, tokenizer, seqs, template, out_dir, fig_dir)


if __name__ == "__main__":
    import argparse

    args = argparse.ArgumentParser()
    args.add_argument("--model_name", type=str, default="llama-3-8b")
    args.add_argument("--seq_fun", type=str, default="happy_negation")
    args.add_argument("--out_dir", type=str, default=None)
    args.add_argument("--fig_dir", type=str, default=None)
    args = args.parse_args()
    main(args.model_name, args.seq_fun, args.out_dir, args.fig_dir)
