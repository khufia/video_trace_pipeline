# Round 01 Plan

## Strategy

Use the summary to anchor the relevant phase after entering the ship's cabin, then localize repeated non-speech background sounds in that interior segment and inspect the bounded clip with dense captioning to infer the likely cause from visible ship motion/context.

## Refinement Instructions

Base the trace on direct evidence from the bounded interior segment after cabin entry, not on the whole-video summary alone. Use step 2 to identify what repeated non-speech sound is actually present and when it occurs. Use step 3 only to infer the cause of that repeated sound from grounded visual context in the same clip: rough seas outside, visible rocking/tilting, human actions, or machinery-related cues. Do not infer engine malfunction unless the clip shows or describes machinery-failure evidence. Do not attribute the sound to human action unless a repeated human-generated action is visibly synchronized with the sound. If the repeated sound is impact-like and coincides with rough seas or vessel rocking, map that carefully to the closest option and explain the evidence-to-option link. If evidence remains ambiguous between wave impact and ship tilting, state the ambiguity rather than overcommitting.

## Steps

1. `visual_temporal_grounder` - Locate the interior ship cabin/bridge segment immediately after the exterior ocean view so later audio analysis is bounded to the correct phase.
   Query: interior of a ship cabin or bridge with a person looking out large windows at a rough ocean
2. `audio_temporal_grounder` - Find the repeated background sound events occurring within the cabin segment after entry.
   Query: repeated background impact or creaking sound inside a ship cabin during rough seas
3. `dense_captioner` - Describe what is happening visually and aurally around the repeated sound events to determine whether the sound is caused by waves, ship tilting, machinery, or human action.

## Files

- [plan json](planner/round_01_plan.json)
- [planner raw](planner/round_01_raw.txt)
