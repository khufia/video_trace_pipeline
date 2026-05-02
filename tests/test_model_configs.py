from pathlib import Path

from video_trace_pipeline.config import load_models_config


def test_main_models_config_uses_planner_and_synthesizer_token_bump():
    repo_root = Path(__file__).resolve().parents[1]
    config = load_models_config(str(repo_root / "configs" / "models.yaml"))

    assert config.agents["planner"].max_tokens == 6000
    assert config.agents["trace_synthesizer"].max_tokens == 16000
    assert "preprocess" not in dict(config.tools["dense_captioner"].extra or {})
    assert config.tools["frame_retriever"].extra["use_reranker"] is False
    assert config.tools["frame_retriever"].extra["device_map"] == "first_two_cuda"


def test_example_tool_server_config_uses_planner_and_synthesizer_token_bump():
    repo_root = Path(__file__).resolve().parents[1]
    config = load_models_config(str(repo_root / "configs" / "models.tool_servers.example.yaml"))

    assert config.agents["planner"].max_tokens == 6000
    assert config.agents["trace_synthesizer"].max_tokens == 16000
    assert config.tools["frame_retriever"].extra["use_reranker"] is False
    assert config.tools["frame_retriever"].extra["device_map"] == "first_two_cuda"
