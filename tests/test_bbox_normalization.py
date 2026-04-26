from pathlib import Path

import pytest
from PIL import Image

from video_trace_pipeline.tool_wrappers.shared import crop_region
from video_trace_pipeline.tool_wrappers.spatial_grounder_runner import _normalize_detections


def test_crop_region_scales_bbox_from_larger_canvas(tmp_path):
    source_path = tmp_path / "frame.png"
    out_path = tmp_path / "crop.png"
    Image.new("RGB", (640, 360), color="white").save(source_path)

    result_path = crop_region(
        str(source_path),
        [207.0, 906.0, 795.0, 977.0],
        out_path,
    )

    assert result_path == str(out_path.resolve())
    with Image.open(Path(result_path)) as cropped:
        assert cropped.size == (196, 24)


def test_crop_region_returns_blank_placeholder_for_outside_bbox(tmp_path):
    source_path = tmp_path / "frame.png"
    out_path = tmp_path / "crop.png"
    Image.new("RGB", (10, 10), color="black").save(source_path)

    result_path = crop_region(
        str(source_path),
        [20.0, 20.0, 30.0, 30.0],
        out_path,
    )

    with Image.open(Path(result_path)) as cropped:
        assert cropped.size == (1, 1)
        assert cropped.getpixel((0, 0)) == (255, 255, 255)


def test_spatial_grounder_normalizes_bbox_to_actual_image_size():
    detections = _normalize_detections(
        [
            {
                "label": "scoreboard",
                "bbox": [207.0, 906.0, 795.0, 977.0],
                "confidence": 0.95,
            }
        ],
        "scoreboard",
        image_size=(640, 360),
    )

    assert detections[0]["bbox"] == pytest.approx([69.0, 302.0, 265.0, 325.6666666666667])
    assert detections[0]["metadata"]["bbox_normalized_to_image"] is True
    assert detections[0]["metadata"]["raw_bbox"] == [207.0, 906.0, 795.0, 977.0]
