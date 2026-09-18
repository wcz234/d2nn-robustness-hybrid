import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from d2nn import detector_contrast, normalized_detector_logits
from perturbations import PerturbationConfig, apply_detector_noise
from robustness_evaluation import evaluate_clean_condition, evaluate_condition, optical_draw_record
from robustness_plan import build_draw_plans, serialized_perturbation_config
from robustness_records import MACRO_F1_CONVENTION, macro_f1_from_confusion


class TinyIndexedDataset(Dataset):
    def __init__(self, indices=None):
        self.indices = list(range(5)) if indices is None else list(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        dataset_index = self.indices[index]
        data = torch.tensor([[[float(dataset_index)]]])
        return data, dataset_index % 3, dataset_index


class RecordingClassifier(torch.nn.Module):
    def __init__(self, feature_count=3):
        super().__init__()
        self.layers = torch.nn.ModuleList([torch.nn.Identity(), torch.nn.Identity()])
        self.size = 4
        self.detector_feature_count = feature_count
        self.anchor = torch.nn.Parameter(torch.tensor(0.0))
        self.optical_fingerprints = []
        self.detector_batches = []

    def forward_with_metrics(self, x, target=None, perturbation_draw=None):
        batch_size = x.shape[0]
        scores = torch.full((batch_size, self.detector_feature_count), 0.25, device=x.device)
        preferred = x.flatten(1)[:, 0].long().remainder(self.detector_feature_count)
        scores.scatter_(1, preferred.unsqueeze(1), 2.0)
        if perturbation_draw is not None:
            record = optical_draw_record(perturbation_draw)
            self.optical_fingerprints.append(record["fingerprint_sha256"])
            shifts = sum(sum(layer.lateral_shift_px) for layer in perturbation_draw.layers)
            scores[:, 0] = (scores[:, 0] + 0.05 * shifts).clamp_min(0.01)
            noise = perturbation_draw.detector_noise_relative
            self.detector_batches.append(None if noise is None else noise.detach().cpu().clone())
            scores = apply_detector_noise(scores, noise)
        result = {"scores": scores, "logits": normalized_detector_logits(scores)}
        if target is not None:
            result["contrast"] = detector_contrast(scores, target)
        return result


class CleanOnlyClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.tensor(0.0))
        self.perturbation_argument_seen = False

    def forward_with_metrics(self, x, target=None, **kwargs):
        self.perturbation_argument_seen = "perturbation_draw" in kwargs
        preferred = x.flatten(1)[:, 0].long().remainder(3)
        logits = torch.full((x.shape[0], 3), -1.0, device=x.device)
        logits.scatter_(1, preferred.unsqueeze(1), 2.0)
        scores = torch.softmax(logits, dim=1)
        return {"scores": scores, "logits": logits, "contrast": detector_contrast(scores, target)}


def make_condition(config, *, draw_count=1, condition_id="test-condition"):
    return {
        "condition_id": condition_id,
        "perturbation_config": serialized_perturbation_config(config),
        "draws": build_draw_plans(
            condition_id=condition_id,
            draw_count=draw_count,
            optical_seed=101,
            detector_seed=202,
        ),
    }


def run_condition(model, loader, condition, *, include_scores=True):
    rows = []
    metrics, summary = evaluate_condition(
        model=model,
        loader=loader,
        device=torch.device("cpu"),
        condition=condition,
        evaluation_id="tiny-evaluation",
        include_scores=include_scores,
        prediction_sink=rows.append,
    )
    return rows, metrics, summary


def test_optical_draw_is_fixed_across_batches_and_rows_are_complete_and_recomputable():
    model = RecordingClassifier()
    loader = DataLoader(TinyIndexedDataset(), batch_size=2, shuffle=False)
    condition = make_condition(
        PerturbationConfig(lateral_shift_max_px=0.5, phase_noise_std_rad=0.1),
        draw_count=2,
    )

    rows, metrics, summary = run_condition(model, loader, condition)

    batches_per_draw = 3
    assert len(rows) == 10
    assert len({(row["condition_id"], row["draw_id"], row["dataset_index"]) for row in rows}) == 10
    assert {row["dataset_index"] for row in rows if row["draw_id"] == 0} == set(range(5))
    assert len(set(model.optical_fingerprints[:batches_per_draw])) == 1
    assert len(set(model.optical_fingerprints[batches_per_draw:])) == 1
    assert model.optical_fingerprints[0] != model.optical_fingerprints[-1]
    for metric in metrics:
        draw_rows = [row for row in rows if row["draw_id"] == metric["draw_id"]]
        assert metric["n_samples"] == len(draw_rows)
        assert metric["n_correct"] == sum(row["correct"] for row in draw_rows)
        assert metric["accuracy"] == metric["n_correct"] / metric["n_samples"]
        assert metric["macro_f1"] == pytest.approx(1.0)
        assert metric["macro_f1_convention"] == MACRO_F1_CONVENTION
        assert metric["confusion_matrix"] == [[2, 0, 0], [0, 2, 0], [0, 0, 1]]
    assert summary["draw_count"] == 2
    assert "not an independent training replicate" in summary["statistical_unit_note"]


def test_macro_f1_is_recomputable_with_zero_division_convention():
    confusion = torch.tensor([[1, 1], [0, 2]])

    assert macro_f1_from_confusion(confusion) == pytest.approx(((2 / 3) + (4 / 5)) / 2)


def test_detector_noise_and_predictions_do_not_depend_on_batch_partition():
    condition = make_condition(
        PerturbationConfig(lateral_shift_max_px=0.2, detector_noise_std_relative=0.25),
        draw_count=2,
    )
    model_two = RecordingClassifier()
    model_three = RecordingClassifier()

    rows_two, metrics_two, _ = run_condition(
        model_two,
        DataLoader(TinyIndexedDataset(), batch_size=2, shuffle=False),
        condition,
    )
    rows_three, metrics_three, _ = run_condition(
        model_three,
        DataLoader(TinyIndexedDataset(), batch_size=3, shuffle=False),
        condition,
    )

    key = lambda row: (row["draw_id"], row["dataset_index"])
    aligned_two = sorted(rows_two, key=key)
    aligned_three = sorted(rows_three, key=key)
    assert [(row["prediction"], row["scores"]) for row in aligned_two] == [
        (row["prediction"], row["scores"]) for row in aligned_three
    ]
    assert [row["detector_noise_sha256"] for row in metrics_two] == [
        row["detector_noise_sha256"] for row in metrics_three
    ]
    assert [row["optical_draw_fingerprint_sha256"] for row in metrics_two] == [
        row["optical_draw_fingerprint_sha256"] for row in metrics_three
    ]


def test_detector_feature_schema_does_not_change_optical_draws():
    condition = make_condition(
        PerturbationConfig(lateral_shift_max_px=0.2, phase_noise_std_rad=0.1, detector_noise_std_relative=0.2),
        draw_count=2,
    )
    _, narrow_metrics, _ = run_condition(
        RecordingClassifier(feature_count=3),
        DataLoader(TinyIndexedDataset(), batch_size=2, shuffle=False),
        condition,
    )
    _, wide_metrics, _ = run_condition(
        RecordingClassifier(feature_count=5),
        DataLoader(TinyIndexedDataset(), batch_size=2, shuffle=False),
        condition,
    )

    assert [row["optical_draw_fingerprint_sha256"] for row in narrow_metrics] == [
        row["optical_draw_fingerprint_sha256"] for row in wide_metrics
    ]
    assert [row["detector_noise_shape"] for row in narrow_metrics] == [[5, 3], [5, 3]]
    assert [row["detector_noise_shape"] for row in wide_metrics] == [[5, 5], [5, 5]]


def test_clean_condition_matches_direct_clean_forward():
    model = RecordingClassifier()
    loader = DataLoader(TinyIndexedDataset(), batch_size=2, shuffle=False)
    rows, metrics, _ = run_condition(model, loader, make_condition(PerturbationConfig()))
    expected = []
    for data, target, indices in loader:
        result = model.forward_with_metrics(data, target=target, perturbation_draw=None)
        for offset, dataset_index in enumerate(indices.tolist()):
            expected.append((dataset_index, int(result["scores"][offset].argmax()), result["scores"][offset].tolist()))

    actual = [(row["dataset_index"], row["prediction"], row["scores"]) for row in rows]
    assert actual == expected
    assert metrics[0]["detector_noise_scope"] == "disabled"
    assert all(row["detector_noise_id"] is None for row in rows)


def test_clean_only_evaluation_reuses_metrics_without_an_optical_draw():
    model = CleanOnlyClassifier()
    loader = DataLoader(TinyIndexedDataset(), batch_size=2, shuffle=False)
    rows = []

    metrics, summary = evaluate_clean_condition(
        model=model,
        loader=loader,
        device=torch.device("cpu"),
        condition=make_condition(PerturbationConfig(), condition_id="clean-only"),
        evaluation_id="clean-evaluation",
        include_scores=True,
        prediction_sink=rows.append,
    )

    assert len(rows) == len(loader.dataset)
    assert metrics[0]["macro_f1"] == pytest.approx(1.0)
    assert metrics[0]["mean_cross_entropy"] > 0
    assert metrics[0]["optical_draw"] is None
    assert metrics[0]["optical_draw_fingerprint_sha256"] is None
    assert summary["draw_count"] == 1
    assert not model.perturbation_argument_seen


def test_clean_only_evaluation_rejects_nonclean_conditions():
    with pytest.raises(ValueError, match="clean-only"):
        evaluate_clean_condition(
            model=CleanOnlyClassifier(),
            loader=DataLoader(TinyIndexedDataset(), batch_size=2, shuffle=False),
            device=torch.device("cpu"),
            condition=make_condition(PerturbationConfig(phase_noise_std_rad=0.1), condition_id="noise"),
            evaluation_id="clean-evaluation",
            include_scores=False,
            prediction_sink=lambda row: None,
        )


def test_duplicate_indices_inside_one_batch_are_rejected():
    loader = DataLoader(TinyIndexedDataset(indices=[0, 0, 2]), batch_size=3, shuffle=False)
    with pytest.raises(ValueError, match="one batch"):
        run_condition(RecordingClassifier(), loader, make_condition(PerturbationConfig()))


def test_shuffled_loader_is_rejected():
    loader = DataLoader(TinyIndexedDataset(), batch_size=2, shuffle=True)
    with pytest.raises(ValueError, match="non-shuffled"):
        run_condition(RecordingClassifier(), loader, make_condition(PerturbationConfig()))
