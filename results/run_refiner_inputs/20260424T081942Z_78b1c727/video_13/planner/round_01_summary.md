# Round 01 Plan

## Strategy

Use the summary-indicated moment around the Ammu-Nation audio to bound ASR, then retrieve frames from that same clip and directly count empty beer bottles visible on the table in the picture.

## Refinement Instructions

Use ASR only to confirm that the relevant audio is the Phil's Ammu Nation line and that the visual evidence should be taken from that same short clip. Base the answer on the retrieved frames, not on the summary alone. In the final trace, explicitly state the counted number of empty beer bottles visible on the table in the picture and map that count to the multiple-choice option. Do not count bottles on the floor, couch area, or elsewhere off the table. If the retrieved frames vary, rely on the clearest frame(s) that actually show the table during the anchored clip.

## Steps

1. `asr` - Verify the exact spoken phrase and anchor the relevant short clip where the line about Phil's Ammu Nation is spoken.
2. `frame_retriever` - Retrieve frames within the same bounded clip that show the living room scene with the passed-out man and the table with empty beer bottles.
   Query: living room scene with a passed-out man on a couch and a table showing empty beer bottles
3. `generic_purpose` - Count how many empty beer bottles are on the table in the retrieved frames from the Ammu-Nation moment.
   Query: In these frames from the moment when the audio says 'Come to Phil's Ammu Nation today', count the number of empty beer bottles visible on the table in the picture. Use only bottles clearly on the table, not bottles elsewhere in the room.

## Files

- [plan json](planner/round_01_plan.json)
- [planner raw](planner/round_01_raw.txt)
