import json

import torch

from diagnose_optical_scale import main


def test_optical_scale_diagnostic_writes_simulation_scoped_json(tmp_path):
    output_path = tmp_path / "diagnostic.json"
    previous_num_threads = torch.get_num_threads()
    previous_deterministic = torch.are_deterministic_algorithms_enabled()

    payload = main(
        [
            "--size",
            "8",
            "--layers",
            "1",
            "--batch-size",
            "10",
            "--num-threads",
            "1",
            "--output",
            str(output_path),
        ]
    )
    persisted = json.loads(output_path.read_text(encoding="utf-8"))

    assert payload["scope"] == "numerical_simulation_only"
    assert persisted["claim_boundary"] == "diagnostic signal scale, not physical-device performance"
    assert persisted["configuration"]["synthetic_input"]["class_counts"] == [1] * 10
    assert set(persisted["presets"]) == {"paper", "d2nn2018_thz"}
    for preset in persisted["presets"].values():
        assert preset["metrics"]["finite_output"]
        assert preset["metrics"]["finite_phase_gradients"]
    assert torch.get_num_threads() == previous_num_threads
    assert torch.are_deterministic_algorithms_enabled() == previous_deterministic
