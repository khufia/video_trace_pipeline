from video_trace_pipeline.schemas import ToolResult
from video_trace_pipeline.tools.extractors import ObservationExtractor


class DummyAtomicizer(object):
    def atomicize(self, source_text: str, context_hint: str = ""):
        assert source_text
        assert context_hint
        return [
            {
                "subject": "scoreboard",
                "subject_type": "entity",
                "predicate": "shows_value",
                "object_text": "10",
                "object_type": "number",
                "atomic_text": "The scoreboard shows 10.",
            }
        ]


def test_asr_extractor_splits_direct_sentences_and_keeps_atomicized_facts():
    extractor = ObservationExtractor(atomicizer=DummyAtomicizer())
    result = ToolResult(
        tool_name="asr",
        data={
            "segments": [
                {
                    "speaker_id": "speaker_1",
                    "text": "Hello there. The score is 10.",
                    "start_s": 1.0,
                    "end_s": 2.5,
                    "confidence": 0.9,
                }
            ]
        },
    )

    observations = extractor.extract(result)

    direct_texts = [item.atomic_text for item in observations if item.predicate == "said"]
    derived_texts = [item.atomic_text for item in observations if item.direct_or_derived == "derived"]

    assert 'speaker_1 said "Hello there." from 1.00s to 2.50s.' in direct_texts
    assert 'speaker_1 said "The score is 10." from 1.00s to 2.50s.' in direct_texts
    assert "The scoreboard shows 10." in derived_texts

    derived = next(item for item in observations if item.atomic_text == "The scoreboard shows 10.")
    assert derived.speaker_id == "speaker_1"
    assert derived.time_start_s == 1.0
    assert derived.time_end_s == 2.5
