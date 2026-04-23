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


def _atomic_chunks(text: str) -> List[str]:
    parts = []
    for chunk in _sentence_chunks(text):
        normalized = re.sub(r"\s+", " ", chunk).strip(" ,;")
        if not normalized:
            continue
        comma_split = [item.strip(" ,;") for item in re.split(r",\s+(?=[A-Za-z0-9])", normalized) if item.strip(" ,;")]
        for piece in comma_split or [normalized]:
            and_split = [item.strip(" ,;") for item in re.split(r"\s+(?:and|but)\s+", piece) if item.strip(" ,;")]
            if len(and_split) > 1:
                parts.extend(and_split)
            else:
                parts.append(piece)
    deduped = []
    seen = set()
    for item in parts:
        lowered = item.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        deduped.append(item)
    return deduped


def evidence_temporal_bounds(observations: List[AtomicObservation]) -> Dict[str, float]:
    time_starts: List[float] = []
    time_ends: List[float] = []
    frame_timestamps: List[float] = []
    for item in list(observations or []):
        if item.time_start_s is not None:
            start_s = float(item.time_start_s)
            end_s = float(item.time_end_s if item.time_end_s is not None else item.time_start_s)
            time_starts.append(start_s)
            time_ends.append(end_s)
        elif item.time_end_s is not None:
            time_ends.append(float(item.time_end_s))
        if item.frame_ts_s is not None:
            frame_timestamps.append(float(item.frame_ts_s))

    payload: Dict[str, float] = {}
    if time_starts or time_ends:
        payload["time_start_s"] = round(min(time_starts or time_ends), 3)
        payload["time_end_s"] = round(max(time_ends or time_starts), 3)
    elif frame_timestamps:
        payload["time_start_s"] = round(min(frame_timestamps), 3)
        payload["time_end_s"] = round(max(frame_timestamps), 3)

    rounded_frames = sorted({round(value, 3) for value in frame_timestamps})
    if len(rounded_frames) == 1:
        payload["frame_ts_s"] = rounded_frames[0]
    return payload


class ObservationExtractor(object):
    VERSION = "v2"

    def __init__(self, atomicizer=None):
        self.atomicizer = atomicizer

    def build_evidence_entry(self, step_id: int, tool_result: ToolResult) -> EvidenceEntry:
        observations = list(getattr(tool_result, "_observations", []) or [])
        observation_ids = [item.metadata.get("observation_id") for item in observations]
        observation_ids = [item for item in observation_ids if item]
        return EvidenceEntry(
            evidence_id="ev_%02d_%s" % (int(step_id), hash_payload({"step": step_id, "tool": tool_result.tool_name}, 8)),
            tool_name=tool_result.tool_name,
            evidence_text=tool_result.summary or tool_result.raw_output_text or "",
            confidence=tool_result.metadata.get("confidence") if isinstance(tool_result.metadata, dict) else None,
            **evidence_temporal_bounds(observations),
            artifact_refs=tool_result.artifact_refs,
            observation_ids=observation_ids,
            metadata={"request_hash": tool_result.request_hash, "cache_hit": tool_result.cache_hit},
        )

    def extract(self, tool_result: ToolResult) -> List[AtomicObservation]:
        tool_name = tool_result.tool_name
        data = dict(tool_result.data or {})
        artifact_ids = [item.artifact_id for item in (tool_result.artifact_refs or []) if getattr(item, "artifact_id", None)]
        observations = []
        if tool_name == "visual_temporal_grounder":
            observations = self._from_temporal(data, audio=False, artifact_ids=artifact_ids)
        elif tool_name == "audio_temporal_grounder":
            observations = self._from_temporal(data, audio=True, artifact_ids=artifact_ids)
        elif tool_name == "frame_retriever":
            observations = self._from_frames(data, artifact_ids=artifact_ids)
        elif tool_name == "asr":
            observations = self._from_asr(data, artifact_ids=artifact_ids)
        elif tool_name == "dense_captioner":
            observations = self._from_dense_caption(data, artifact_ids=artifact_ids)
        elif tool_name == "ocr":
            observations = self._from_ocr(data, artifact_ids=artifact_ids)
        elif tool_name == "spatial_grounder":
            observations = self._from_spatial(data, artifact_ids=artifact_ids)
        elif tool_name == "generic_purpose":
            observations = self._from_generic(data, artifact_ids=artifact_ids)
        else:
            observations = self._fallback(data, tool_name, artifact_ids=artifact_ids)
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

    def _llm_atomicize(
        self,
        tool_name: str,
        text: str,
        context_hint: str,
        *,
        default_subject: str,
        default_subject_type: str,
        time_start_s: Optional[float] = None,
        time_end_s: Optional[float] = None,
        frame_ts_s: Optional[float] = None,
        source_artifact_refs: Optional[List[str]] = None,
        speaker_id: Optional[str] = None,
    ) -> List[AtomicObservation]:
        if self.atomicizer is None:
            return []
        source_text = str(text or "").strip()
        if not source_text:
            return []
        try:
            facts = self.atomicizer.atomicize(source_text, context_hint=context_hint)
        except Exception:
            return []
        observations = []
        for fact in facts:
            atomic_text = str(fact.get("atomic_text") or "").strip()
            if not atomic_text:
                continue
            observations.append(
                self._make_observation(
                    tool_name,
                    subject=str(fact.get("subject") or default_subject).strip() or default_subject,
                    subject_type=str(fact.get("subject_type") or default_subject_type).strip() or default_subject_type,
                    predicate=str(fact.get("predicate") or "reported").strip() or "reported",
                    object_text=str(fact.get("object_text") or "").strip(),
                    object_type=str(fact.get("object_type") or "text").strip() or "text",
                    atomic_text=atomic_text,
                    time_start_s=time_start_s,
                    time_end_s=time_end_s,
                    frame_ts_s=frame_ts_s,
                    source_artifact_refs=source_artifact_refs,
                    speaker_id=speaker_id,
                    direct_or_derived="derived",
                )
            )
        return observations

    def _from_temporal(self, data: Dict[str, Any], audio: bool = False, artifact_ids: Optional[List[str]] = None) -> List[AtomicObservation]:
        query = str(data.get("query") or "").strip() or ("audio event" if audio else "visual event")
        clips = list(data.get("clips") or [])
        items = []
        for clip in clips:
            start = float(clip.get("start_s", clip.get("start", 0.0)) or 0.0)
            end = float(clip.get("end_s", clip.get("end", start)) or start)
            confidence = clip.get("confidence")
            if confidence is None and isinstance(clip.get("metadata"), dict):
                confidence = clip.get("metadata", {}).get("confidence")
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
                    source_artifact_refs=artifact_ids,
                )
            )
        return items

    def _from_frames(self, data: Dict[str, Any], artifact_ids: Optional[List[str]] = None) -> List[AtomicObservation]:
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
                    confidence=frame.get("metadata", {}).get("relevance_score") or frame.get("relevance_score"),
                    source_artifact_refs=artifact_ids,
                )
            )
        return items

    def _from_asr(self, data: Dict[str, Any], artifact_ids: Optional[List[str]] = None) -> List[AtomicObservation]:
        def _segment_observations(segment: Dict[str, Any], clip: Optional[Dict[str, Any]] = None) -> List[AtomicObservation]:
            speaker = str(segment.get("speaker_id") or segment.get("speaker") or "unknown_speaker")
            text = str(segment.get("text") or "").strip()
            clip = dict(clip or {})
            start = float(segment.get("start_s", segment.get("start", clip.get("start_s", 0.0))) or 0.0)
            end = float(segment.get("end_s", segment.get("end", start)) or start)
            if not text:
                return []

            items = []
            sentence_parts = _sentence_chunks(text) or [text]
            for part in sentence_parts:
                chunk = str(part or "").strip()
                if not chunk:
                    continue
                items.append(
                    self._make_observation(
                        "asr",
                        subject=speaker,
                        subject_type="speaker",
                        predicate="said",
                        object_text=chunk,
                        object_type="utterance",
                        atomic_text='%s said "%s" from %.2fs to %.2fs.' % (speaker, chunk, start, end),
                        time_start_s=start,
                        time_end_s=end,
                        speaker_id=speaker,
                        confidence=segment.get("confidence"),
                        source_artifact_refs=artifact_ids,
                    )
                )

            derived = self._llm_atomicize(
                "asr",
                text,
                context_hint="ASR transcript from %.2fs to %.2fs by %s." % (start, end, speaker),
                default_subject=speaker,
                default_subject_type="speaker",
                time_start_s=start,
                time_end_s=end,
                source_artifact_refs=artifact_ids,
                speaker_id=speaker,
            )
            if derived:
                items.extend(derived)
            return items

        items = []
        transcripts = list(data.get("transcripts") or [])
        if transcripts:
            for transcript in transcripts:
                clip = dict(transcript.get("clip") or {})
                for segment in list(transcript.get("segments") or []):
                    items.extend(_segment_observations(segment, clip=clip))
            return items
        for segment in list(data.get("segments") or []):
            items.extend(_segment_observations(segment))
        return items

    def _from_dense_caption(self, data: Dict[str, Any], artifact_ids: Optional[List[str]] = None) -> List[AtomicObservation]:
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
                            source_artifact_refs=artifact_ids,
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
                            source_artifact_refs=artifact_ids,
                        )
                    )
            for text in _atomic_chunks(caption.get("on_screen_text", "")):
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
                        source_artifact_refs=artifact_ids,
                    )
                )
            visual_observations = self._llm_atomicize(
                "dense_captioner",
                caption.get("visual", ""),
                context_hint="Dense caption visual text from %.2fs to %.2fs." % (start, end),
                default_subject="video_segment",
                default_subject_type="clip",
                time_start_s=start,
                time_end_s=end,
                source_artifact_refs=artifact_ids,
            )
            if visual_observations:
                items.extend(visual_observations)
            for visual_fact in ([] if visual_observations else _atomic_chunks(caption.get("visual", ""))):
                items.append(
                    self._make_observation(
                        "dense_captioner",
                        subject="video_segment",
                        subject_type="clip",
                        predicate="visual_fact",
                        object_text=visual_fact,
                        object_type="text",
                        atomic_text="Visual fact from %.2fs to %.2fs: %s." % (start, end, visual_fact),
                        time_start_s=start,
                        time_end_s=end,
                        source_artifact_refs=artifact_ids,
                    )
                )
            audio_observations = self._llm_atomicize(
                "dense_captioner",
                caption.get("audio", ""),
                context_hint="Dense caption audio text from %.2fs to %.2fs." % (start, end),
                default_subject="audio_track",
                default_subject_type="audio",
                time_start_s=start,
                time_end_s=end,
                source_artifact_refs=artifact_ids,
            )
            if audio_observations:
                items.extend(audio_observations)
            for audio_fact in ([] if audio_observations else _atomic_chunks(caption.get("audio", ""))):
                if audio_fact.lower() in {"none", "unknown"}:
                    continue
                items.append(
                    self._make_observation(
                        "dense_captioner",
                        subject="audio_track",
                        subject_type="audio",
                        predicate="contains_audio_event",
                        object_text=audio_fact,
                        object_type="text",
                        atomic_text="Audio fact from %.2fs to %.2fs: %s." % (start, end, audio_fact),
                        time_start_s=start,
                        time_end_s=end,
                        source_artifact_refs=artifact_ids,
                    )
                )
            for attribute in caption.get("attributes") or []:
                attr_text = str(attribute or "").strip()
                if not attr_text:
                    continue
                items.append(
                    self._make_observation(
                        "dense_captioner",
                        subject="video_segment",
                        subject_type="clip",
                        predicate="has_attribute",
                        object_text=attr_text,
                        object_type="attribute",
                        atomic_text="An observed attribute from %.2fs to %.2fs is: %s." % (start, end, attr_text),
                        time_start_s=start,
                        time_end_s=end,
                        source_artifact_refs=artifact_ids,
                    )
                )
        return items

    def _from_ocr(self, data: Dict[str, Any], artifact_ids: Optional[List[str]] = None) -> List[AtomicObservation]:
        items = []
        reads = list(data.get("reads") or [])
        if reads:
            for read in reads:
                frame = dict(read.get("frame") or {})
                region = dict(read.get("region") or {})
                timestamp = read.get("timestamp_s")
                if timestamp is None and frame:
                    timestamp = frame.get("timestamp_s")
                lines = read.get("lines") or []
                full_text = str(read.get("text") or "").strip()
                if isinstance(lines, str):
                    lines = [line for line in _sentence_chunks(lines)]
                if not lines and full_text:
                    lines = [line for line in _sentence_chunks(full_text)]
                for line in lines:
                    if isinstance(line, dict):
                        text = str(line.get("text") or "").strip()
                        bbox = line.get("bbox") or region.get("bbox")
                    else:
                        text = str(line or "").strip()
                        bbox = region.get("bbox") if region else None
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
                            frame_ts_s=float(timestamp) if timestamp is not None else None,
                            source_artifact_refs=artifact_ids,
                        )
                    )
            return items
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
                    source_artifact_refs=artifact_ids,
                )
            )
        return items

    def _from_spatial(self, data: Dict[str, Any], artifact_ids: Optional[List[str]] = None) -> List[AtomicObservation]:
        items = []
        groundings = list(data.get("groundings") or [])
        if groundings:
            for grounding in groundings:
                frame = dict(grounding.get("frame") or {})
                frame_ts = grounding.get("timestamp")
                if frame_ts is None:
                    frame_ts = frame.get("timestamp_s") or frame.get("timestamp")
                for detection in grounding.get("detections") or []:
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
                            frame_ts_s=float(frame_ts) if frame_ts is not None else None,
                            confidence=float(confidence) if confidence is not None else None,
                            source_artifact_refs=artifact_ids,
                        )
                    )
            return items
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
                    source_artifact_refs=artifact_ids,
                )
            )
        return items

    def _from_generic(self, data: Dict[str, Any], artifact_ids: Optional[List[str]] = None) -> List[AtomicObservation]:
        chunks = []
        for item in data.get("supporting_points") or []:
            chunks.extend(_atomic_chunks(item))
        text = str(data.get("answer") or data.get("response") or data.get("text") or "").strip()
        llm_chunks = self._llm_atomicize(
            "generic_purpose",
            " ".join([str(item) for item in data.get("supporting_points") or []] + ([text] if text else [])),
            context_hint="Generic-purpose evidence response.",
            default_subject="generic_tool",
            default_subject_type="tool",
            source_artifact_refs=artifact_ids,
        )
        if llm_chunks:
            return llm_chunks
        if text:
            chunks.extend(_atomic_chunks(text))
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
                source_artifact_refs=artifact_ids,
            )
            for chunk in chunks
        ]

    def _fallback(self, data: Dict[str, Any], tool_name: str, artifact_ids: Optional[List[str]] = None) -> List[AtomicObservation]:
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
                source_artifact_refs=artifact_ids,
            )
        ]
