"""Tests for scripts.detection_common."""
import pytest
import torch

# Import after conftest has added paths
from detection_common import (
    COCO_INSTANCE_CATEGORY_NAMES,
    get_class_ids_for_names,
    draw_detections,
    detections_to_coco_style,
    load_model,
    run_inference,
)


def test_coco_names_nonempty():
    assert len(COCO_INSTANCE_CATEGORY_NAMES) > 80
    assert COCO_INSTANCE_CATEGORY_NAMES[0] == "__background__"
    assert "person" in COCO_INSTANCE_CATEGORY_NAMES
    assert "car" in COCO_INSTANCE_CATEGORY_NAMES


def test_get_class_ids_for_names():
    ids = get_class_ids_for_names(["person", "car"])
    assert 1 in ids  # person
    assert 3 in ids  # car
    ids2 = get_class_ids_for_names(["person", "PERSON", " car "])
    assert ids == ids2
    assert get_class_ids_for_names([]) == set()
    assert get_class_ids_for_names(["notaclass"]) == set()


def test_draw_detections_no_crash():
    """draw_detections should not crash with dummy tensors (in-place draw on frame)."""
    import cv2
    import numpy as np
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    boxes = torch.tensor([[10.0, 10.0, 50.0, 50.0], [60.0, 60.0, 90.0, 90.0]])
    scores = torch.tensor([0.9, 0.7])
    labels = torch.tensor([1, 3])  # person, car
    draw_detections(frame, boxes, scores, labels, score_threshold=0.5)
    # Just check we didn't crash; frame is modified in place
    assert frame.shape == (100, 100, 3)


def test_draw_detections_with_class_filter():
    import cv2
    import numpy as np
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    boxes = torch.tensor([[10.0, 10.0, 50.0, 50.0], [60.0, 60.0, 90.0, 90.0]])
    scores = torch.tensor([0.9, 0.7])
    labels = torch.tensor([1, 3])
    draw_detections(
        frame, boxes, scores, labels,
        score_threshold=0.5,
        class_filter={1},  # only person
    )
    assert frame.shape == (100, 100, 3)


def test_detections_to_coco_style():
    boxes = torch.tensor([[10.0, 20.0, 50.0, 60.0]])
    scores = torch.tensor([0.95])
    labels = torch.tensor([1])
    out = detections_to_coco_style(0, 100, 100, boxes, scores, labels, score_threshold=0.5)
    assert len(out) == 1
    assert out[0]["image_id"] == 0
    assert out[0]["category_id"] == 1
    assert out[0]["bbox"] == [10.0, 20.0, 40.0, 40.0]
    assert abs(out[0]["score"] - 0.95) < 1e-5


def test_load_model_cpu():
    device = torch.device("cpu")
    # Use v1 to avoid downloading new weights if only v1 is cached
    model = load_model(device, model_type="fasterrcnn_resnet50_fpn")
    assert next(model.parameters()).device == device


def test_run_inference_cpu():
    """Run inference on a tiny tensor (CPU) to ensure model runs."""
    device = torch.device("cpu")
    model = load_model(device, model_type="fasterrcnn_resnet50_fpn")
    # 3x32x32 RGB
    x = torch.rand(3, 32, 32).to(device)
    out = run_inference(model, x, device, use_fp16=False)
    assert "boxes" in out and "scores" in out and "labels" in out
    assert out["boxes"].shape[0] == out["scores"].shape[0] == out["labels"].shape[0]
