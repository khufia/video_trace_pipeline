# Round 01 Plan

## Strategy

Use the narrated temporal anchor about Scrubby's first writing attempt, then inspect the bounded clip visually to determine whether the first blue-pen drawing is poor handwriting, neat handwriting, a picture with seams, or no writing.

## Refinement Instructions

Base the trace on the earliest relevant localized blue-pen writing attempt, not on later multicolor drawing scenes. Use the visual evidence from the retrieved frames to decide among the four options. Preserve the temporal distinction from the summary: the first blue-pen attempt occurs before the later discussion of drawing pictures with seams. If the frames show shaky or malformed written marks, map that to poor handwriting; do not confuse later 'picture with seams' evidence with this earlier first drawing/writing attempt. If the retrieved frames include multiple candidate moments, the trace should rely on the earliest one that clearly shows blue-pen marks being made.

## Steps

1. `visual_temporal_grounder` - Localize the moment where Scrubby first uses a blue pen to write or draw on the board, corresponding to the early poor-writing demonstration before later drawing experiments.
   Query: Scrubby using a blue pen or blue marker to make its first writing marks on a board, showing the initial poor writing attempt
2. `frame_retriever` - Retrieve representative frames from the localized candidate clip so the quality and type of the first blue-pen marks can be inspected directly.
   Query: Scrubby making blue pen marks on the board, showing whether the marks are handwriting, a picture with seams, or just movement without writing
3. `generic_purpose` - Classify what the first blue-pen drawing looks like from the retrieved frames, focusing on the answer-distinguishing visual evidence.
   Query: From these frames, determine how Scrubby's first blue-pen drawing appears: poor handwriting, neat clear handwriting, a picture with visible seams between sections, or no writing at all with only movement. Identify which description best matches the earliest relevant frames.

## Files

- [plan json](planner/round_01_plan.json)
- [planner raw](planner/round_01_raw.txt)
