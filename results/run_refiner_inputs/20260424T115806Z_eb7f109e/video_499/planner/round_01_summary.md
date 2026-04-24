# Round 01 Plan

## Strategy

Use one visual branch to localize the binoculars scene and inspect frames for which hand holds the pot, and one speech branch to transcribe the family conversation about whether Sheldon's brother was premature.

## Refinement Instructions

Build the answer from direct evidence only. For the hand subquestion, rely on step 3's frame-based determination from the binoculars scene; do not infer handedness from summary text alone. If multiple frames differ, use only the frames where Dr. Sturgis and the pot are clearly visible and note any ambiguity rather than guessing. For the premature-baby subquestion, use the ASR transcript from step 4 to capture the explicit dialogue: the brother asks about being premature and the mother says he was born premature, even if the exchange is comedic or contradictory. Map the final answer to the option combining the verified hand and whether the dialogue explicitly says yes or no about prematurity.

## Steps

1. `asr` - Transcribe the dialogue where Sheldon's brother asks whether he was born before the wedding and the mother discusses prematurity.
2. `visual_temporal_grounder` - Localize the scene where Sheldon watches Dr. Sturgis through binoculars and Dr. Sturgis is visible holding a pot.
   Query: Sheldon looking through binoculars out a window at Dr. Sturgis outside near Meemaw's house, with Dr. Sturgis visible holding a pot
3. `frame_retriever` - Retrieve the most relevant frames within the localized binoculars scene that clearly show Dr. Sturgis and the pot for hand identification.
   Query: Dr. Sturgis holding a pot while being observed through binoculars, showing which hand holds the pot
4. `generic_purpose` - Determine whether Dr. Sturgis is holding the pot in his left or right hand from the retrieved frames.
   Query: For each frame, identify Dr. Sturgis and the pot, then determine whether the pot is being held in Dr. Sturgis's left hand or right hand. Use only clearly visible frames and report the consistent hand if visible.

## Files

- [plan json](planner/round_01_plan.json)
- [planner raw](planner/round_01_raw.txt)
