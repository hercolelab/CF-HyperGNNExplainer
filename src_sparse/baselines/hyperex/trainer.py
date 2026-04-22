import os
import random

import torch

from utils import get_hyper_neighbourhood_fast, normalize_propagation

from baselines.hyperex.attention import Attention
from baselines.hyperex.common import (
    build_local_to_global,
    compute_hyperedge_embeddings_global,
    extract_induced_edge_global_ids,
    local_class_probabilities,
    normalize_propagation_dense,
    sparse_incidence_to_dense,
    training_soft_weights_from_alpha,
)
from baselines.hyperex.infonce import InfoNCE_loss


def rand_train_test_idx(
    labels: torch.Tensor,
    train_prop: float = 0.5,
    valid_prop: float = 0.25,
    seed: int = 0,
) -> dict[str, torch.Tensor]:
    num_nodes = int(labels.size(0))
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    perm = torch.randperm(num_nodes, generator=generator)

    train_num = int(num_nodes * train_prop)
    valid_num = int(num_nodes * valid_prop)
    valid_end = min(train_num + valid_num, num_nodes)

    return {
        "train": perm[:train_num],
        "valid": perm[train_num:valid_end],
        "test": perm[valid_end:],
    }


def load_attention_checkpoint(
    attention_module: torch.nn.Module,
    checkpoint_path: str | None,
    device: torch.device,
    strict: bool = True,
) -> torch.nn.Module:
    if checkpoint_path is None:
        return attention_module
    if not os.path.exists(checkpoint_path):
        if strict:
            raise FileNotFoundError(
                f"Attention checkpoint not found: {checkpoint_path}"
            )
        return attention_module
    state_dict = torch.load(checkpoint_path, map_location=device)
    attention_module.load_state_dict(state_dict)
    return attention_module


def save_attention_checkpoint(
    attention_module: torch.nn.Module, checkpoint_path: str
) -> None:
    out_dir = os.path.dirname(os.path.abspath(checkpoint_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    torch.save(attention_module.state_dict(), checkpoint_path)


def build_attention_module(
    num_classes: int,
    device: torch.device,
    checkpoint_path: str | None = None,
    strict: bool = True,
) -> Attention:
    attention_module = Attention(embed_dim=num_classes).to(device)
    return load_attention_checkpoint(
        attention_module, checkpoint_path, device, strict=strict
    )


def _yield_minibatches(node_idxs: list[int], batch_size: int):
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    for start in range(0, len(node_idxs), batch_size):
        yield node_idxs[start : start + batch_size]


def train_hyperex_attention(
    model: torch.nn.Module,
    H: torch.Tensor,
    features: torch.Tensor,
    labels: torch.Tensor,
    n_hops: int,
    thresh_num: int,
    epochs: int,
    lr: float,
    train_prop: float,
    valid_prop: float,
    node_samples: int | None,
    seed: int,
    device: torch.device,
    attention_module: torch.nn.Module | None = None,
    batch_size: int = 64,
    tau: float = 1.0,
) -> torch.nn.Module:
    H = H.coalesce().to(device)
    features = features.to(device)
    labels = labels.to(device)

    torch.manual_seed(seed)
    random.seed(seed)

    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    if attention_module is None:
        num_classes = int(labels.max().item()) + 1
        attention_module = Attention(embed_dim=num_classes).to(device)
    else:
        attention_module = attention_module.to(device)

    optimiser = torch.optim.Adam(attention_module.parameters(), lr=lr)
    split_idx = rand_train_test_idx(
        labels=labels,
        train_prop=train_prop,
        valid_prop=valid_prop,
        seed=seed,
    )

    node_idxs = split_idx["train"].tolist()
    if node_samples is not None:
        node_idxs = node_idxs[:node_samples]

    with torch.no_grad():
        S_full = normalize_propagation(H)
        global_z = model(features, S_full, return_embeddings=True)
        h_global = compute_hyperedge_embeddings_global(H, global_z)

    for epoch in range(epochs):
        attention_module.train()

        random.shuffle(node_idxs)
        batch_losses = []
        valid_batches = 0

        for batch_nodes in _yield_minibatches(node_idxs, batch_size):
            optimiser.zero_grad()

            full_graph_embeds = []
            pruned_embeds = []

            for node_idx in batch_nodes:
                comp_H, sub_feat, _sub_labels, node_dict = get_hyper_neighbourhood_fast(
                    node_idx=node_idx,
                    H=H,
                    n_hops=n_hops,
                    features=features,
                    labels=labels,
                )

                if node_idx not in node_dict:
                    continue

                comp_H = comp_H.coalesce().to(device)
                sub_feat = sub_feat.to(device)

                local_to_global = build_local_to_global(node_dict, device)
                H_dense = sparse_incidence_to_dense(comp_H)
                if H_dense.sum() == 0:
                    continue

                z_local = local_class_probabilities(global_z, local_to_global)
                comp_edge_global_ids = extract_induced_edge_global_ids(H, node_dict)
                h_edges = h_global[comp_edge_global_ids]
                target_local = int(node_dict[node_idx])

                alpha, _ = attention_module.forward_dense(z_local, h_edges, H_dense)
                if alpha.sum() == 0:
                    continue

                dense_w = training_soft_weights_from_alpha(alpha, H_dense)
                S_local = normalize_propagation_dense(dense_w)
                out_local = model(sub_feat, S_local, return_embeddings=True)

                if torch.any(torch.isnan(out_local)):
                    print(f"NaN encountered for node {node_idx}; skipping.")
                    continue

                pruned_embeds.append(out_local[target_local])
                full_graph_embeds.append(global_z[node_idx])

            if len(pruned_embeds) < 2:
                continue

            full_batch = torch.stack(full_graph_embeds, dim=0)
            pruned_batch = torch.stack(pruned_embeds, dim=0)
            loss = InfoNCE_loss(full_batch, pruned_batch, tau=tau, normalize=False)
            loss.backward()
            optimiser.step()

            batch_losses.append(loss.item())
            valid_batches += 1

        if not batch_losses:
            print(f"Epoch {epoch + 1}/{epochs}: no valid training batches.")
            continue

        mean_loss = sum(batch_losses) / len(batch_losses)
        print(
            f"Epoch {epoch + 1}/{epochs}: loss={mean_loss:.6f} "
            f"(valid_batches={valid_batches})"
        )

    attention_module.eval()
    return attention_module
