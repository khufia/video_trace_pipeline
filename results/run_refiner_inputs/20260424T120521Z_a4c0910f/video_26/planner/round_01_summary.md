# Round 01 Plan

## Strategy

Use the caption summary to identify that the question refers to the in-movie character Pando/Panda, then ground the relevant spoken description directly with ASR over the full short video because the answer choices are character traits likely stated in dialogue rather than requiring broad visual search.

## Refinement Instructions

Build the trace around directly grounded transcript evidence about the movie character Pando/Panda. Use the ASR output to find lines where other characters describe or criticize how Pando is portrayed in the movie. Preserve the distinction between the in-movie character and viewers commenting on that portrayal. Do not rely on the summary alone for the final claim. If the transcript explicitly supports one option, cite that wording. If the transcript shows traits like clumsy, on the phone, or unable to swim but none of the listed options match exactly, the trace should state that mismatch clearly rather than guessing from plausibility. Only use visual details if they are explicitly needed to disambiguate between bow tie, glasses, broom, or Einstein and the transcript is insufficient.

## Steps

1. `asr` - Transcribe the full short video to capture dialogue describing Pando/Panda's traits and any explicit mentions that distinguish the answer options.

## Files

- [plan json](planner/round_01_plan.json)
- [planner raw](planner/round_01_raw.txt)
