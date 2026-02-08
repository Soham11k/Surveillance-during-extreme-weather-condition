"""Integration tests: run detection scripts and check outputs."""
import json
import tempfile
from pathlib import Path

import pytest

# Paths set by conftest
from detection_runner import run_detection_image


def test_run_detection_image_produces_file():
    """Run detection on a real image and assert output image and optional JSON exist."""
    project_root = Path(__file__).resolve().parent.parent
    sample = project_root / "sample_images" / "frame_0000.png"
    if not sample.exists():
        pytest.skip("sample_images/frame_0000.png not found")
    with tempfile.TemporaryDirectory() as tmp:
        out_img = Path(tmp) / "out_det.jpg"
        out_json = Path(tmp) / "out.json"
        run_detection_image(
            image_path=str(sample),
            output_path=str(out_img),
            score_threshold=0.5,
            save_json_path=str(out_json),
        )
        assert out_img.is_file()
        assert out_img.stat().st_size > 0
        assert out_json.is_file()
        data = json.loads(out_json.read_text())
        assert "annotations" in data
        assert "width" in data and "height" in data
