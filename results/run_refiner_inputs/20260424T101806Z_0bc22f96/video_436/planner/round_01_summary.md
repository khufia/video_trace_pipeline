# Round 01 Plan

## Strategy

Localize the red-sauce usage moment, then inspect that bounded clip for distinct audio events associated with squeezing/using the ketchup bottle so the trace can count the sounds directly rather than infer from the summary.

## Refinement Instructions

Use step 1 only as the temporal anchor for the ketchup-use moment; do not treat clip ranking as evidence of chronology. Build the trace from direct evidence in steps 2 and 3 about audible events during use of the red sauce. Count only distinct sounds that occur when the ketchup bottle is used or squeezed, not general dinner-table speech or reactions unless they are clearly part of the bottle-use sound sequence. Prefer the bounded audio evidence from step 3 for the count, and use step 2 to clarify whether multiple retrieved events are actually different sound types or repeats of the same sound. If the evidence shows one bottle-use sound before the product change and a different bottle-use sound after the product change, make that comparison explicit. Map the verified count to the closest answer option and avoid relying on the whole-video summary alone.

## Steps

1. `visual_temporal_grounder` - Find the clip where the red sauce (Heinz ketchup) is being used or squeezed at the table.
   Query: person using or squeezing a Heinz ketchup bottle at the Thanksgiving dinner table
2. `audio_temporal_grounder` - Retrieve distinct non-speech sound events within the localized ketchup-use clip to support counting different sounds.
   Query: distinct non-speech sounds made when squeezing or using the Heinz ketchup bottle at the dinner table
3. `dense_captioner` - Describe the localized ketchup-use clip with emphasis on audible events and bottle use so the trace can identify how many different sounds occur.

## Files

- [plan json](planner/round_01_plan.json)
- [planner raw](planner/round_01_raw.txt)
