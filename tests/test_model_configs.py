from pathlib import Path

from video_trace_pipeline.config import load_models_config


def test_main_models_config_uses_planner_and_synthesizer_token_bump():
    repo_root = Path(__file__).resolve().parents[1]
    config = load_models_config(str(repo_root / "configs" / "models.yaml"))

    assert config.agents["planner"].max_tokens == 6000
    assert config.agents["trace_synthesizer"].max_tokens == 16000
    dense_preprocess = dict(config.tools["dense_captioner"].extra["preprocess"])
    assert dense_preprocess["enabled"] is False
    assert config.tools["visual_temporal_grounder"].extra["device_map"] is None
    assert config.tools["frame_retriever"].extra["use_reranker"] is False
    assert config.tools["frame_retriever"].extra["device_map"] is None


def test_example_tool_server_config_uses_planner_and_synthesizer_token_bump():
    repo_root = Path(__file__).resolve().parents[1]
    config = load_models_config(str(repo_root / "configs" / "models.tool_servers.example.yaml"))

    assert config.agents["planner"].max_tokens == 6000
    assert config.agents["trace_synthesizer"].max_tokens == 16000
    dense_preprocess = dict(config.tools["dense_captioner"].extra["preprocess"])
    assert dense_preprocess["enabled"] is False
    assert config.tools["visual_temporal_grounder"].extra["device_map"] is None
    assert config.tools["frame_retriever"].extra["use_reranker"] is False
    assert config.tools["frame_retriever"].extra["device_map"] is None
