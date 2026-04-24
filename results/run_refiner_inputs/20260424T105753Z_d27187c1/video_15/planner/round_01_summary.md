# Round 01 Plan

## Strategy

Use ASR directly because the question is about who wants to send the body down the water slide, and the summary already indicates the decisive evidence is spoken dialogue and voting near the end.

## Refinement Instructions

Base the trace primarily on spoken evidence, not visual guesswork. Use the ASR transcript to quote or paraphrase the decisive lines about the water-slide idea and the later vote. The key distinction is not just who mentions the idea first, but who is said to want it and who wins the vote. If speaker attribution is noisy, rely on the transcript content and the role descriptions from the summary only as supporting context, not as sole evidence. The final trace should explain which group or person is explicitly counted as wanting to do it and then select the matching option.

## Steps

1. `asr` - Transcribe the full video dialogue so the trace can identify who proposes the slide idea, who supports it, and who is counted in the final vote.
2. `generic_purpose` - Interpret the transcript to determine which answer option matches the people who want to send the body into the water slide.
   Query: From the transcript, identify who wants to send the body down the water slide. Distinguish between the person who first gets blamed for suggesting it, the people who explicitly support it, and the final vote count. Map the result to these options only: the person in black, the person in red, the children, or nobody wants.

## Files

- [plan json](planner/round_01_plan.json)
- [planner raw](planner/round_01_raw.txt)
