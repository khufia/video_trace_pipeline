from video_trace_pipeline.cli.main import _build_direct_task, _parse_options, _parse_tool_names


def test_parse_options_supports_json_and_delimited_text():
    assert _parse_options('["A", "B"]') == ["A", "B"]
    assert _parse_options("A || B || C") == ["A", "B", "C"]


def test_parse_tool_names_supports_aliases_and_dedupes():
    assert _parse_tool_names('["temporal_grounder", "generic", "generic_purpose"]') == [
        "visual_temporal_grounder",
        "generic_purpose",
    ]


def test_build_direct_task_creates_adhoc_task(tmp_path):
    video_path = tmp_path / "demo.mp4"
    video_path.write_bytes(b"video")
    task = _build_direct_task(
        video_path=str(video_path),
        question="What happens in the clip?",
        options_text='["A", "B"]',
    )
    assert task.benchmark == "adhoc"
    assert task.video_id == "demo"
    assert task.options == ["A", "B"]
    assert task.sample_key.startswith("demo__")
