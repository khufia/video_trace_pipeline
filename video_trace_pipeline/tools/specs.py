from __future__ import annotations

LOCAL_PROCESS_IMPLEMENTATION = "local_process"
LOCAL_ASR_IMPLEMENTATION = "local_whisperx_v2"
CUSTOM_IMPLEMENTATION = "custom_extension"

_TOOL_IMPLEMENTATIONS = {
    "visual_temporal_grounder": LOCAL_PROCESS_IMPLEMENTATION,
    "frame_retriever": LOCAL_PROCESS_IMPLEMENTATION,
    "audio_temporal_grounder": LOCAL_PROCESS_IMPLEMENTATION,
    "asr": LOCAL_ASR_IMPLEMENTATION,
    "dense_captioner": LOCAL_PROCESS_IMPLEMENTATION,
    "ocr": LOCAL_PROCESS_IMPLEMENTATION,
    "spatial_grounder": LOCAL_PROCESS_IMPLEMENTATION,
    "generic_purpose": LOCAL_PROCESS_IMPLEMENTATION,
}


def tool_implementation(tool_name: str) -> str:
    normalized = str(tool_name or "").strip()
    return _TOOL_IMPLEMENTATIONS.get(normalized, CUSTOM_IMPLEMENTATION)


def uses_process_wrapper(tool_name: str) -> bool:
    return tool_implementation(tool_name) == LOCAL_PROCESS_IMPLEMENTATION
