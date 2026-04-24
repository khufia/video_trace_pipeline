# Round 01 Plan

## Strategy

Localize the metallic bang in audio, then inspect the corresponding visual moment on the bridge to determine the man's posture/action among the options.

## Refinement Instructions

Base the trace on the audio-localized metallic bang window from step 1, not on the summary alone. Use the retrieved frames from step 2 as the visual evidence for the same moment. In the final trace, describe only the man's observable posture/action at the bang moment and map it directly to the closest answer option among walking, drinking, leaning, or standing. Do not infer from general scene context or from earlier/later moments outside the localized bang clip. If the retrieved frames show slight variation across the clip, prefer the frame(s) closest to the center of the bang event and state any residual uncertainty briefly rather than overcommitting.

## Steps

1. `audio_temporal_grounder` - Find the time window where a loud metallic bang occurs.
   Query: loud metallic bang sound
2. `frame_retriever` - Retrieve frames from the localized bang clip showing the man on the ship bridge at the moment of the sound.
   Query: man on the ship bridge near the windows, showing what the man is doing at the moment of the metallic bang
3. `generic_purpose` - Determine whether the man is walking, drinking, leaning, or standing in the retrieved frames from the bang moment.
   Query: For each provided frame, identify the man's action or posture on the bridge and determine which option best matches: walking, drinking, leaning, or standing.

## Files

- [plan json](planner/round_01_plan.json)
- [planner raw](planner/round_01_raw.txt)
