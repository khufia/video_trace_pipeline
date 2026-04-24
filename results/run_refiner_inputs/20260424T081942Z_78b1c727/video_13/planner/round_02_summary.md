# Round 02 Plan

## Strategy

Preserve the already grounded temporal anchor and bottle count, and run one focused visual follow-up on the previously retrieved frame bundle to determine whether the two table bottles are empty rather than merely visible.

## Refinement Instructions

Keep the prior supported facts that the relevant audio is the Ammu-Nation line around 129s-134s and that two beer bottles are visible on the glass table in the anchored scene. Replace the unsupported claim that the answer is definitely two empty bottles unless the new frame analysis directly supports emptiness. The repaired trace should explicitly distinguish 'two visible beer bottles' from 'two empty beer bottles.' Use step 2 as the controlling evidence for the missing attribute: if both bottles are visibly empty, then the trace may conclude 2 and map to option C; if only one is visibly empty, map to option B; if none are visibly empty, map to option A; if emptiness remains indeterminate from the frames, the trace must state that the evidence does not fully support a definite multiple-choice selection rather than repeating the old overconfident answer.

## Steps

1. `frame_retriever` - Retrieve a tight frame bundle within the already anchored Ammu-Nation moment, optimized for clearly seeing the two bottles on the glass table and their fill state.
   Query: glass table in the living room scene with two beer bottles, showing whether the bottles contain liquid or are empty
2. `generic_purpose` - Inspect the retrieved frames and determine for each visible bottle on the table whether it appears empty, partially full, or indeterminate, then count how many are empty.
   Query: For the bottles sitting on the glass table in these frames, determine whether each bottle appears empty, partially full, full, or indeterminate based only on visible liquid level or transparency cues. Then report how many table bottles are visibly empty and whether the evidence is decisive.

## Files

- [plan json](planner/round_02_plan.json)
- [planner raw](planner/round_02_raw.txt)
