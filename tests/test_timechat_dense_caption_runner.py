from video_trace_pipeline.tool_wrappers import timechat_dense_caption_runner


def test_extract_json_list_items_salvages_complete_prefix_from_truncated_output():
    raw = """
[
  {"timestamp": "00:00-00:10", "segment_detail_caption": "first"},
  {"timestamp": "00:10-00:20", "segment_detail_caption": "second"},
  {"timestamp": "00:20-00:30", "segment_detail_caption": "third
""".strip()

    parsed = timechat_dense_caption_runner._extract_json_list_items(raw)

    assert parsed == [
        {"timestamp": "00:00-00:10", "segment_detail_caption": "first"},
        {"timestamp": "00:10-00:20", "segment_detail_caption": "second"},
    ]


def test_normalize_span_maps_timechat_native_fields_into_pipeline_schema():
    span = timechat_dense_caption_runner._normalize_span(
        {
            "timestamp": "00:10-00:12",
            "segment_detail_caption": "A chart titled 'FASTEST GROWING GROCERS IN 2022' appears.",
            "speech_content": "Aldi is the fastest-growing grocer by store count in the country.",
            "acoustics_content": "Upbeat instrumental music continues.",
            "camera_state": "Static full shot.",
            "video_background": "Light green infographic background.",
            "storyline": "The segment emphasizes Aldi's growth.",
            "shooting_style": "Animated infographic.",
        },
        start_s=0.0,
        end_s=30.0,
    )

    assert span["start"] == 10.0
    assert span["end"] == 12.0
    assert span["visual"] == "A chart titled 'FASTEST GROWING GROCERS IN 2022' appears."
    assert "speech: Aldi is the fastest-growing grocer by store count in the country." in span["audio"]
    assert "acoustics: Upbeat instrumental music continues." in span["audio"]
    assert span["on_screen_text"] == "FASTEST GROWING GROCERS IN 2022"
    assert "camera_state: Static full shot." in span["attributes"]
    assert "video_background: Light green infographic background." in span["attributes"]
    assert "storyline: The segment emphasizes Aldi's growth." in span["attributes"]
    assert "shooting_style: Animated infographic." in span["attributes"]


def test_summary_from_captions_uses_mapped_fields():
    summary = timechat_dense_caption_runner._summary_from_captions(
        [
            {
                "visual": "Milk, eggs, and bread appear on a white table.",
                "audio": "speech: Milk is $2.18.",
                "on_screen_text": "PRICE IN NEW YORK CITY",
                "actions": [],
                "objects": [],
                "attributes": ["camera_state: top-down"],
            }
        ]
    )

    assert "Milk, eggs, and bread appear on a white table." in summary
    assert "speech: Milk is $2.18." in summary
    assert "PRICE IN NEW YORK CITY" in summary
    assert "camera_state: top-down" in summary
