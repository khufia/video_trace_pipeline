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


def test_execute_payload_skips_sampled_frames_when_collection_disabled(monkeypatch, tmp_path):
    sample_calls = {"count": 0}

    def _fake_sample_request_frames(*args, **kwargs):
        del args, kwargs
        sample_calls["count"] += 1
        return [{"frame_path": "unused.jpg", "timestamp_s": 0.0}]

    class _FakeRunner(object):
        def generate(self, conversation, max_new_tokens):
            del conversation, max_new_tokens
            return '{"captions":[{"start":0.0,"end":5.0,"visual":"A shopper enters the store.","audio":"","on_screen_text":"ALDI","actions":["enters store"],"objects":["shopper"],"attributes":["bright aisle"]}],"overall_summary":"A shopper enters the store."}'

    class _ClipContext(object):
        def __enter__(self):
            return str(tmp_path / "clip.mp4")

        def __exit__(self, exc_type, exc, tb):
            del exc_type, exc, tb
            return False

    monkeypatch.setattr(timechat_dense_caption_runner, "sample_request_frames", _fake_sample_request_frames)
    monkeypatch.setattr(timechat_dense_caption_runner, "resolve_model_path", lambda *args, **kwargs: "/models/fake")
    monkeypatch.setattr(timechat_dense_caption_runner, "resolved_device_label", lambda runtime: "cuda:0")
    monkeypatch.setattr(timechat_dense_caption_runner, "scratch_dir", lambda runtime, name: tmp_path / name)
    monkeypatch.setattr(timechat_dense_caption_runner, "extracted_clip", lambda *args, **kwargs: _ClipContext())
    monkeypatch.setattr(timechat_dense_caption_runner, "make_timechat_video_conversation", lambda *args, **kwargs: [])

    payload = {
        "request": {
            "clip": {"video_id": "video1", "start_s": 0.0, "end_s": 5.0},
            "granularity": "segment",
            "focus_query": "",
        },
        "task": {"video_path": str(tmp_path / "video.mp4")},
        "runtime": {
            "model_name": "fake-model",
            "device": "cuda:0",
            "extra": {
                "fps": 1.0,
                "max_frames": 96,
                "collect_sampled_frames": False,
                "use_audio_in_video": False,
                "max_new_tokens": 700,
            },
        },
    }

    result = timechat_dense_caption_runner.execute_payload(payload, runner=_FakeRunner())

    assert sample_calls["count"] == 0
    assert result["sampled_frames"] == []
    assert result["captions"][0]["visual"] == "A shopper enters the store."
