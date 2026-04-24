# Round 02 Plan

## Strategy

Repair the prior transcript-only trace by grounding the in-movie visual appearance of Pando/Panda at the moments where the movie shows that character, then directly check which option matches the visible character design. Use the summary and prior ASR-derived anchors to target the movie-within-the-video scenes rather than re-running broad ASR.

## Refinement Instructions

Preserve the earlier transcript-supported context that the question refers to the in-movie character Pando/Panda, not a random panda elsewhere in the video. Replace the unsupported elimination-by-ASR reasoning with direct visual evidence from the retrieved Pando/Panda frames. The updated trace should use step 3 as the controlling evidence for the multiple-choice selection. If the frames clearly show one option, select that option and explicitly reject the others only insofar as they are visually contradicted or not present in the grounded frames. Do not rely on transcript absence to rule out glasses, bow tie, broom, or Einstein. If multiple retrieved frames correspond to different appearances of the same in-movie character, compare them and base the answer on the appearance that is actually shown for Pando/Panda in the movie scenes. If the new visual evidence contradicts the prior tentative reasoning, update the final answer accordingly.

## Steps

1. `visual_temporal_grounder` - Localize clips where the in-movie character Pando/Panda is visibly shown so the answer can be based on direct visual evidence rather than transcript absence.
   Query: clips showing the in-movie character Pando or Panda on screen, especially the sidekick character in the spy movie and the panda character being criticized after the screening
2. `frame_retriever` - Retrieve representative frames from the localized Pando/Panda clips that clearly show the character's body and accessories for option checking.
   Query: Pando or Panda character on screen, showing clothing, accessories, and overall identity clearly
3. `generic_purpose` - Determine which of the answer options is visually supported by the retrieved frames and whether the character wears glasses, wears a bow tie, is depicted as a broom, or is depicted as Einstein.
   Query: For the character Pando or Panda shown in these frames, identify which option is directly supported by visible evidence: wearing glasses, wearing a bow tie, being a broom, or being Einstein. State which options are visibly present or absent based only on the frames.

## Files

- [plan json](planner/round_02_plan.json)
- [planner raw](planner/round_02_raw.txt)
