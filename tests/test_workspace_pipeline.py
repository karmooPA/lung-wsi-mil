"""Exercise workspace routing and guards around the existing scientific stages."""

import json

import pytest

from lung_wsi_mil.workflows import workspace


@pytest.mark.parametrize("name", ["../archive", "CON", "folder/subfolder"])
def test_unsafe_run_names_are_rejected_before_creating_output(tmp_path, name):
    with pytest.raises(SystemExit) as error:
        workspace.main(["--data-root", str(tmp_path), "--run-name", name, "--check-only"])
    assert error.value.code == 2
    assert not (tmp_path / "output").exists()


def test_raw_check_routes_new_derivatives_away_from_accepted_artifacts(
    tmp_path, monkeypatch, capsys
):
    checked = tmp_path / "derived" / "patches"
    checked.mkdir(parents=True)
    marker = checked / "keep"
    marker.write_text("accepted", encoding="utf-8")
    monkeypatch.setattr(workspace, "check_sources", lambda *args: None)
    assert (
        workspace.main(
            ["--data-root", str(tmp_path), "--run-name", "qa", "--start-at", "raw", "--check-only"]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    assert result["slides"] == 694
    assert result["seeds"] == [2, 3, 5, 7, 11]
    assert result["heatmaps_per_seed"] == {"2": 141, "3": 139, "5": 138, "7": 141, "11": 140}
    assert result["paths"]["patches"] == str(tmp_path / "derived" / "runs" / "qa" / "patches")
    assert marker.read_text(encoding="utf-8") == "accepted"
    assert not (tmp_path / "output").exists()
    assert not (tmp_path / "derived" / "runs").exists()
