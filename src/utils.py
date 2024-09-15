import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import umap


def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def visualize_train_result(log_dir):
    df = pd.read_csv(os.path.join(log_dir, 'metrics.csv'))
    df = df.drop(['train_loss',], axis=1)
    df = df.set_index(['epoch', 'step'])
    df = df.fillna(0)
    df = df.groupby(level=[0, 1]).sum()
    df = df.reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_title('Training Result')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Metrics Value')
    ax.set_ylim(0, 1)
    ax.grid(True)
    df[df.columns].plot(ax=ax, label=df.columns)
    plt.legend()
    plt.savefig(os.path.join(log_dir, 'train_result.png'))
    plt.close()


def visualize_logits_umap(data_loader, model, reducer, output_file):
    logits_list = []
    targets_list = []

    for x, t in data_loader:
        x = x.to(model.device)
        emb = model.get_embedding(x)
        logits_list.append(emb.detach().cpu().numpy())
        targets_list.append(t.numpy())

    logits_array = np.vstack(logits_list)
    targets_array = np.concatenate(targets_list)


    if reducer is None:
        n_neighbors = 15 if len(logits_array) > 15 else len(logits_array) - 1
        reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=0.1, n_components=2, random_state=42, n_jobs=1)
        embedding = reducer.fit_transform(logits_array, y=targets_array)
    else:
        embedding = reducer.transform(logits_array)

    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=targets_array, s=5)
    plt.title('UMAP projection of the Logits, colored by target')
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    plt.colorbar(scatter)
    plt.savefig(output_file)
    plt.close()
    return reducer
