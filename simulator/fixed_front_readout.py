"""Train readouts on frozen, paired optical/control representations.

One source checkpoint is one optical training replicate. Different heads or
random/control fronts derived from it are not additional independent replicates.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, TensorDataset, random_split

from perturbations import LayerPerturbation, PerturbationDraw
from robustness_io import artifact_record, finalize_staged_output, staged_output_directory, write_json
from robustness_records import macro_f1_from_confusion, tensor_sha256
from robustness_target import load_robustness_target
from tasks import build_classification_transform, get_classification_dataset_config
from training_provenance import _index_sha256, environment_manifest, file_sha256


def phase_identity(model):
    return tensor_sha256(torch.stack([layer.phase.detach().cpu() for layer in model.layers]))


def frozen_front(source, kind, seed):
    if kind not in ("trained", "random", "free", "input"):
        raise ValueError(f"unknown front {kind}")
    model = deepcopy(source).eval()
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        if kind == "random":
            for layer in model.layers:
                layer.phase.copy_(torch.rand(layer.phase.shape, generator=generator) * (2 * np.pi))
        elif kind == "free":
            for layer in model.layers:
                layer.phase.zero_()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def nominal_draw(model, manifest, path):
    if path == "latent":
        return None
    configuration = manifest.get("training_perturbations", {})
    phase_config = configuration.get("nominal_phase_config")
    if phase_config is None:
        phase_config = {"phase_quantization_levels": configuration.get("config", {}).get("phase_quantization_levels"), "phase_filter_std_px": 0.0}
    return PerturbationDraw(
        tuple(LayerPerturbation() for _ in model.layers),
        phase_quantization_levels=phase_config["phase_quantization_levels"],
        phase_filter_std_px=phase_config["phase_filter_std_px"],
    )


@torch.no_grad()
def extract_features(model, loader, *, kind, pools, draw=None):
    buckets = {f"pool{pool}": [] for pool in pools}
    if kind == "trained":
        buckets["detector"] = []
    labels = []
    before = phase_identity(model)
    for inputs, targets in loader:
        inputs = inputs.to(next(model.parameters()).device)
        intensity = model._embed_input(inputs).abs().square() if kind == "input" else model.output_intensity(inputs, draw)
        for pool in pools:
            buckets[f"pool{pool}"].append(F.adaptive_avg_pool2d(intensity.unsqueeze(1), (pool, pool)).flatten(1).cpu())
        if kind == "trained":
            buckets["detector"].append(model.detect_from_intensity(intensity).cpu())
        labels.append(targets.cpu())
    if phase_identity(model) != before:
        raise RuntimeError("frozen optical weights changed during feature extraction")
    if not labels:
        raise ValueError("feature loader must not be empty")
    return {name: torch.cat(values) for name, values in buckets.items()}, torch.cat(labels)


def normalize_features(features):
    return features / features.mean(1, keepdim=True).clamp_min(1e-8)


def head_model(feature_count, kind, hidden_dim=32):
    if kind == "linear":
        return nn.Linear(feature_count, 10)
    if kind == "mlp":
        return nn.Sequential(nn.Linear(feature_count, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 10))
    raise ValueError(f"unknown head {kind}")


@torch.no_grad()
def metrics(head, features, labels, *, device="cpu", batch_size=512):
    confusion = torch.zeros(10, 10, dtype=torch.long)
    ce_total = 0.0
    head.eval()
    for start in range(0, len(labels), batch_size):
        target = labels[start:start + batch_size].to(device)
        logits = head(normalize_features(features[start:start + batch_size]).to(device))
        predictions = logits.argmax(1)
        ce_total += float(F.cross_entropy(logits, target, reduction="sum"))
        confusion += torch.bincount((target * 10 + predictions).cpu(), minlength=100).reshape(10, 10)
    return {"accuracy": float(confusion.diag().sum() / len(labels)), "macro_f1": macro_f1_from_confusion(confusion), "mean_cross_entropy": ce_total / len(labels), "confusion_matrix": confusion.tolist()}


def fit_readout(train_features, train_labels, val_features, val_labels, *, kind, seed, epochs, learning_rate, batch_size, hidden_dim, device):
    if epochs < 1 or learning_rate <= 0 or batch_size < 1:
        raise ValueError("head fit requires positive epochs, learning rate and batch size")
    torch.manual_seed(seed)
    head = head_model(train_features.shape[1], kind, hidden_dim).to(device)
    loader = DataLoader(TensorDataset(normalize_features(train_features), train_labels), batch_size=batch_size, shuffle=True, generator=torch.Generator().manual_seed(seed))
    optimizer = torch.optim.Adam(head.parameters(), lr=learning_rate)
    best_key, best_state, best_epoch, history = (-float("inf"), -float("inf")), None, None, []
    for epoch in range(1, epochs + 1):
        head.train()
        loss_sum = 0.0
        for features, labels in loader:
            optimizer.zero_grad()
            loss = F.cross_entropy(head(features.to(device)), labels.to(device))
            if not torch.isfinite(loss):
                raise ValueError("non-finite readout training loss")
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * len(labels)
        validation = metrics(head, val_features, val_labels, device=device)
        history.append({"epoch": epoch, "train_cross_entropy": loss_sum / len(train_labels), "validation_accuracy": validation["accuracy"], "validation_cross_entropy": validation["mean_cross_entropy"]})
        key = (validation["accuracy"], -validation["mean_cross_entropy"])
        if key > best_key:
            best_key, best_epoch = key, epoch
            best_state = {name: value.detach().cpu().clone() for name, value in head.state_dict().items()}
    head.load_state_dict(best_state)
    return head, {"best_epoch": best_epoch, "best_validation_accuracy": best_key[0], "history": history}


def _subset(dataset, limit):
    if limit is not None:
        if limit < 1 or limit > len(dataset):
            raise ValueError("subset limit must be within the corresponding split")
        return Subset(dataset, list(range(limit)))
    return dataset


def run(args):
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    if args.num_threads < 1 or args.batch_size < 1 or args.epochs < 1 or args.lr <= 0 or args.hidden_dim < 1:
        raise ValueError("thread count, batch size, epochs, learning rate and hidden dimension must be positive")
    torch.set_num_threads(args.num_threads)
    device = torch.device(args.device)
    target = load_robustness_target(args.checkpoint, device=device)
    if target.model_variant not in ("d2nn", "hybrid") or target.model.activation_type != "none":
        raise ValueError("frozen-front controls require a phase-only optical classifier")
    if len(set(args.pools)) != len(args.pools) or min(args.pools) < 1 or max(args.pools) > target.model.size:
        raise ValueError("pools must be distinct positive sizes within the optical grid")
    configuration = get_classification_dataset_config(target.dataset_key)
    transform = build_classification_transform(configuration)
    source_data = configuration["dataset_cls"](args.data_dir, train=True, download=True, transform=transform)
    split = target.manifest["data_split"]
    train_set, val_set = random_split(source_data, [split["train_samples"], split["validation_samples"]], generator=torch.Generator().manual_seed(split["split_seed"]))
    for field, subset in [("train_indices_sha256", train_set), ("validation_indices_sha256", val_set)]:
        if _index_sha256(subset.indices) != split[field]:
            raise ValueError("source checkpoint split hashes do not match reconstructed data")
    selected_train, selected_val = _subset(train_set, args.train_limit), _subset(val_set, args.validation_limit)
    loaders = [DataLoader(dataset, batch_size=args.batch_size, shuffle=False) for dataset in (selected_train, selected_val)]
    source_hash = phase_identity(target.model)
    frozen_models, trained_heads, records = {}, {}, []
    with staged_output_directory(output) as stage:
        for kind in args.fronts:
            model = frozen_front(target.model, kind, target.training_seed + 10000)
            frozen_models[kind] = model
            draw = nominal_draw(model, target.manifest, args.phase_path)
            (train_features, train_labels), (val_features, val_labels) = [extract_features(model, loader, kind=kind, pools=args.pools, draw=draw) for loader in loaders]
            for representation in train_features:
                for head_kind in args.heads:
                    name = f"{kind}_{representation}_{head_kind}"
                    head, history = fit_readout(train_features[representation], train_labels, val_features[representation], val_labels, kind=head_kind, seed=target.training_seed, epochs=args.epochs, learning_rate=args.lr, batch_size=args.batch_size, hidden_dim=args.hidden_dim, device=device)
                    checkpoint = stage / f"{name}.pth"
                    torch.save(head.state_dict(), checkpoint)
                    trained_heads[name] = head
                    records.append({"method_id": name, "front": kind, "representation": representation, "head": head_kind, "source_training_seed": target.training_seed, "readout_seed": target.training_seed, "front_phase_sha256": phase_identity(model), "feature_count": train_features[representation].shape[1], "head_parameters": sum(p.numel() for p in head.parameters()), "head_checkpoint": artifact_record(checkpoint), **history})
                    print(f"{name}: best validation accuracy={history['best_validation_accuracy']:.4f}", flush=True)
        # All model selection is complete before the test set is constructed.
        test_set = configuration["dataset_cls"](args.data_dir, train=False, download=True, transform=transform)
        selected_test = _subset(test_set, args.test_limit)
        test_loader = DataLoader(selected_test, batch_size=args.batch_size, shuffle=False)
        fixed_detector = None
        for kind, model in frozen_models.items():
            draw = nominal_draw(model, target.manifest, args.phase_path)
            test_features, labels = extract_features(model, test_loader, kind=kind, pools=args.pools, draw=draw)
            for record in records:
                if record["front"] == kind:
                    record["test_metrics"] = metrics(trained_heads[record["method_id"]], test_features[record["representation"]], labels, device=device)
            if kind == "trained":
                predictions = test_features["detector"].argmax(1)
                confusion = torch.bincount(labels * 10 + predictions, minlength=100).reshape(10, 10)
                fixed_detector = {"accuracy": float((predictions == labels).float().mean()), "macro_f1": macro_f1_from_confusion(confusion), "confusion_matrix": confusion.tolist()}
        if phase_identity(target.model) != source_hash:
            raise RuntimeError("source checkpoint phase weights were modified")
        payload = {
            "schema_version": "fixed-front-readout/v1", "scope": "numerical_simulation_only",
            "evidence_status": "subset_pilot_not_main_result" if any(v is not None for v in (args.train_limit, args.validation_limit, args.test_limit)) else "full_data_single_source_seed_requires_cross_source_seed_summary",
            "source_checkpoint_sha256": target.checkpoint_sha256, "source_training_seed": target.training_seed,
            "source_phase_sha256_before": source_hash, "source_phase_sha256_after": phase_identity(target.model),
            "source_validation_phase_config": target.manifest.get("training_perturbations", {}).get("nominal_phase_config"),
            "phase_path": args.phase_path, "paired_random_front_seed": target.training_seed + 10000,
            "sample_scope": {"train_samples": len(selected_train), "validation_samples": len(selected_val), "test_samples": len(selected_test), "full_train_split_sha256": split["train_indices_sha256"], "full_validation_split_sha256": split["validation_indices_sha256"], "train_limit": args.train_limit, "validation_limit": args.validation_limit, "test_limit": args.test_limit},
            "statistics": {"independent_unit": "source_training_seed", "head_variants_are_not_independent_replicates": True},
            "feature_semantics": "mean-normalized intensity; input control uses the SAME embedded amplitude image squared; free front retains propagation geometry with zero phase; random phase is uniform on [0,2pi)",
            "training_objective": "readout cross entropy; optical weights frozen; same head initialization and shuffle seed at each compatible dimension",
            "fixed_detector_test_metrics": fixed_detector, "records": records,
            "configuration": vars(args),
            "source_sha256": file_sha256(Path(__file__)),
            "environment": environment_manifest(device, Path(__file__).parent),
        }
        write_json(stage / "readout_results.json", payload)
        finalize_staged_output(stage, output)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--data-dir", default=str(Path(__file__).parent / "data"))
    parser.add_argument("--fronts", nargs="+", choices=["trained", "random", "free", "input"], default=["trained", "random", "free", "input"])
    parser.add_argument("--heads", nargs="+", choices=["linear", "mlp"], default=["linear", "mlp"])
    parser.add_argument("--pools", type=int, nargs="+", default=[4, 8])
    parser.add_argument("--phase-path", choices=["latent", "validation"], default="latent")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--num-threads", type=int, default=2)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--validation-limit", type=int)
    parser.add_argument("--test-limit", type=int)
    args = parser.parse_args(argv)
    if len(set(args.fronts)) != len(args.fronts) or len(set(args.heads)) != len(args.heads):
        parser.error("fronts and heads must be unique")
    print(f"Wrote {run(args)}")


if __name__ == "__main__":
    main()
