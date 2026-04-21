from __future__ import annotations

import json
import re
from typing import Any, Dict, Iterable, List, Optional

from ..common import hash_payload
from ..schemas import AtomicObservation, EvidenceEntry, ToolResult


def _sentence_chunks(text: str) -> List[str]:
    chunks = []
    for part in re.split(r"[;\n]+", str(text or "")):
        for sentence in re.split(r"(?<=[.!?])\s+", part):
            cleaned = str(sentence or "").strip()
            if cleaned:
                chunks.append(cleaned)
    return chunks


class ObservationExtractor(object):
    def build_evidence_entry(self, step_id: int, tool_result: ToolResult) -> EvidenceEntry:
        observation_ids = [item.metadata.get("observation_id") for item in getattr(tool_result, "_observations", [])]
        observation_ids = [item for item in observation_ids if item]
        return EvidenceEntry(
            evidence_id="ev_%02d_%s" % (int(step_id), hash_payload({"step": step_id, "tool": tool_result.tool_name}, 8)),
            tool_name=tool_result.tool_name,
            evidence_text=tool_result.summary or tool_result.raw_output_text or "",
            confidence=tool_result.metadata.get("confidence") if isinstance(tool_result.metadata, dict) else None,
            artifact_refs=tool_result.artifact_refs,
            observation_ids=observation_ids,
            metadata={"request_hash": tool_result.request_hash, "cache_hit": tool_result.cache_hit},
        )

    def extract(self, tool_result: ToolResult) -> List[AtomicObservation]:
        tool_name = tool_result.tool_name
        data = dict(tool_result.data or {})
        observations = []
        if tool_name == "visual_temporal_grounder":
            observations = self._from_temporal(data, audio=False)
        elif tool_name == "audio_temporal_grounder":
            observations = self._from_temporal(data, audio=True)
        elif tool_name == "frame_retriever":
            observations = self._from_frames(data)
        elif tool_name == "asr":
            observations = self._from_asr(data)
        elif tool_name == "dense_captioner":
            observations = self._from_dense_caption(data)
        elif tool_name == "ocr":
            observations = self._from_ocr(data)
        elif tool_name == "spatial_grounder":
            observations = self._from_spatial(data)
        elif tool_name == "generic_purpose":
            observations = self._from_generic(data)
        else:
            observations = self._fallback(data, tool_name)
        for item in observations:
            item.metadata["observation_id"] = item.observation_id
        tool_result.metadata["observation_ids"] = [item.observation_id for item in observations]
        return observations

    def _make_observation(
        self,
        tool_name: str,
        subject: str,
        subject_type: str,
        predicate: str,
        atomic_text: str,
        object_text: str = "",
        object_type: str = "text",
        **kwargs
    ) -> AtomicObservation:
        payload = {
            "tool_name": tool_name,
            "subject": subject,
            "predicate": predicate,
            "atomic_text": atomic_text,
            "object_text": object_text,
            "time_start_s": kwargs.get("time_start_s"),
            "time_end_s": kwargs.get("time_end_s"),
            "frame_ts_s": kwargs.get("frame_ts_s"),
        }
        return AtomicObservation(
            observation_id="obs_%s" % hash_payload(payload, 16),
            subject=subject,
            subject_type=subject_type,
            predicate=predicate,
            object_text=object_text,
            object_type=object_type,
            numeric_value=kwargs.get("numeric_value"),
            unit=kwargs.get("unit"),
            time_start_s=kwargs.get("time_start_s"),
            time_end_s=kwargs.get("time_end_s"),
            frame_ts_s=kwargs.get("frame_ts_s"),
            bbox=kwargs.get("bbox"),
            speaker_id=kwargs.get("speaker_id"),
            confidence=kwargs.get("confidence"),
            source_tool=tool_name,
            source_artifact_refs=list(kwargs.get("source_artifact_refs") or []),
            direct_or_derived=kwargs.get("direct_or_derived", "direct"),
            atomic_text=atomic_text,
            metadata={},
        )

    def _from_temporal(self, data: Dict[str, Any], audio: bool = False) -> List[AtomicObservation]:
        query = str(data.get("query") or "").strip() or ("audio event" if audio else "visual event")
        clips = list(data.get("clips") or [])
        items = []
        for clip in clips:
            start = float(clip.get("start_s", clip.get("start", 0.0)) or 0.0)
            end = float(clip.get("end_s", clip.get("end", start)) or start)
            confidence = clip.get("confidence")
            items.append(
                self._make_observation(
                    "audio_temporal_grounder" if audio else "visual_temporal_grounder",
                    subject=query,
                    subject_type="event_query",
                    predicate="present_in_interval",
                    object_text="%.2fs-%.2fs" % (start, end),
                    object_type="time_interval",
                    atomic_text='"%s" is present from %.2fs to %.2fs.' % (query, start, end),
                    time_start_s=start,
                    time_end_s=end,
                    confidence=float(confidence) if confidence is not None else None,
                )
            )
        return items

    def _from_frames(self, data: Dict[str, Any]) -> List[AtomicObservation]:
        items = []
        for frame in list(data.get("frames") or []):
            ts = float(frame.get("timestamp_s", frame.get("timestamp", 0.0)) or 0.0)
            items.append(
                self._make_observation(
                    "frame_retriever",
                    subject=str(frame.get("query") or "requested frame"),
                    subject_type="frame_query",
                    predicate="retrieved_frame_at",
                    object_text="%.2fs" % ts,
                    object_type="timestamp",
                    atomic_text="A candidate frame was retrieved at %.2fs." % ts,
                    frame_ts_s=ts,
                )
            )
        return items

    def _from_asr(self, data: Dict[str, Any]) -> List[AtomicObservation]:
        items = []
        for segment in list(data.get("segments") or []):
            speaker = str(segment.get("speaker_id") or segment.get("speaker") or "unknown_speaker")
            text = str(segment.get("text") or "").strip()
            start = float(segment.get("start_s", segment.get("start", 0.0)) or 0.0)
            end = float(segment.get("end_s", segment.get("end", start)) or start)
            if not text:
                continue
            items.append(
                self._make_observation(
                    "asr",
                    subject=speaker,
                    subject_type="speaker",
                    predicate="said",
                    object_text=text,
                    object_type="utterance",
                    atomic_text='%s said "%s" from %.2fs to %.2fs.' % (speaker, text, start, end),
                    time_start_s=start,
                    time_end_s=end,
                    speaker_id=speaker,
                    confidence=segment.get("confidence"),
                )
            )
        return items

    def _from_dense_caption(self, data: Dict[str, Any]) -> List[AtomicObservation]:
        items = []
        for caption in list(data.get("captions") or []):
            start = float(caption.get("start", 0.0) or 0.0)
            end = float(caption.get("end", start) or start)
            for action in caption.get("actions") or []:
                action_text = str(action or "").strip()
                if action_text:
                    items.append(
                        self._make_observation(
                            "dense_captioner",
                            subject="video_segment",
                            subject_type="clip",
                            predicate="shows_action",
                            object_text=action_text,
                            object_type="action",
                            atomic_text='The clip from %.2fs to %.2fs shows: %s.' % (start, end, action_text),
                            time_start_s=start,
                            time_end_s=end,
                        )
                    )
            for obj in caption.get("objects") or []:
                obj_text = str(obj or "").strip()
                if obj_text:
                    items.append(
                        self._make_observation(
                            "dense_captioner",
                            subject=obj_text,
                            subject_type="entity",
                            predicate="appears_in_clip",
                            object_text="%.2fs-%.2fs" % (start, end),
                            object_type="time_interval",
                            atomic_text="%s appears from %.2fs to %.2fs." % (obj_text, start, end),
                            time_start_s=start,
                            time_end_s=end,
                        )
                    )
            for text in _sentence_chunks(caption.get("on_screen_text", "")):
                if text.lower() == "none":
                    continue
                items.append(
                    self._make_observation(
                        "dense_captioner",
                        subject="screen",
                        subject_type="frame",
                        predicate="shows_text",
                        object_text=text,
                        object_type="text",
                        atomic_text='Visible text from %.2fs to %.2fs: "%s".' % (start, end, text),
                        time_start_s=start,
                        time_end_s=end,
                    )
                )
        return items

    def _from_ocr(self, data: Dict[str, Any]) -> List[AtomicObservation]:
        items = []
        lines = data.get("lines") or data.get("text_lines") or []
        if isinstance(lines, str):
            lines = [line for line in _sentence_chunks(lines)]
        full_text = str(data.get("text") or data.get("full_text") or "").strip()
        if not lines and full_text:
            lines = [line for line in _sentence_chunks(full_text)]
        for line in lines:
            if isinstance(line, dict):
                text = str(line.get("text") or "").strip()
                bbox = line.get("bbox")
            else:
                text = str(line or "").strip()
                bbox = None
            if not text:
                continue
            items.append(
                self._make_observation(
                    "ocr",
                    subject="screen",
                    subject_type="frame",
                    predicate="contains_text",
                    object_text=text,
                    object_type="text",
                    atomic_text='OCR detected text: "%s".' % text,
                    bbox=bbox,
                )
            )
        return items

    def _from_spatial(self, data: Dict[str, Any]) -> List[AtomicObservation]:
        items = []
        detections = data.get("detections") or []
        for detection in detections:
            label = str(detection.get("label") or detection.get("name") or "entity").strip()
            bbox = detection.get("bbox") or detection.get("box")
            confidence = detection.get("confidence")
            items.append(
                self._make_observation(
                    "spatial_grounder",
                    subject=label,
                    subject_type="entity",
                    predicate="located_in_frame",
                    object_text=str(bbox),
                    object_type="bbox",
                    atomic_text="%s is located at %s in the frame." % (label, bbox),
                    bbox=bbox,
                    frame_ts_s=data.get("timestamp_s") or data.get("timestamp"),
                    confidence=float(confidence) if confidence is not None else None,
                )
            )
        return items

    def _from_generic(self, data: Dict[str, Any]) -> List[AtomicObservation]:
        text = str(data.get("answer") or data.get("response") or data.get("text") or "").strip()
        return [
            self._make_observation(
                "generic_purpose",
                subject="generic_tool",
                subject_type="tool",
                predicate="reported",
                object_text=chunk,
                object_type="text",
                atomic_text=chunk,
                direct_or_derived="derived",
            )
            for chunk in _sentence_chunks(text)
        ]

    def _fallback(self, data: Dict[str, Any], tool_name: str) -> List[AtomicObservation]:
        text = json.dumps(data, ensure_ascii=False)
        return [
            self._make_observation(
                tool_name,
                subject=tool_name,
                subject_type="tool",
                predicate="reported",
                object_text=text[:500],
                object_type="json",
                atomic_text=text[:500],
                direct_or_derived="derived",
            )
        ]
