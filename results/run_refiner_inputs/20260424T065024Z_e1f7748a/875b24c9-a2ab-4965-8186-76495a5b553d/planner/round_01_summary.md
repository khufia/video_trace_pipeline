# Round 01 Plan

## Strategy

Localize the customer-survey chart/table comparing grocers on Store Cleanliness and Value for Dollar, retrieve the best frames from that bounded clip, then use targeted frame analysis to read the two percentages for Walmart, Target, Whole Foods, and Albertsons and identify the largest gap.

## Refinement Instructions

Build the trace around the survey graphic evidence, not the narration alone. Use step 1 only to justify the temporal anchor. Use step 2 to establish the exact frames that contain the comparison graphic. Use step 3 as the decisive evidence source: preserve the extracted percentages for Store Cleanliness and Value for Dollar for Walmart, Target, Whole Foods, and Albertsons, and explicitly compare the absolute gaps across those four stores. The final answer should be based on the largest verified gap and then mapped to the closest option if the extracted difference is approximate. Do not infer from Aldi-related narration, since Aldi is not among the answer choices. If multiple retrieved frames show partial or progressive versions of the graphic, rely on the frame(s) that most clearly show both categories and the relevant store rows; do not average across inconsistent partial reads.

## Steps

1. `visual_temporal_grounder` - Find the clip where the survey graphic or chart shows grocers with customer-rated categories including Store Cleanliness and Value for Dollar.
   Query: graphic or chart comparing grocery stores on customer survey ratings, including Store Cleanliness and Value for Dollar for stores such as Walmart, Target, Whole Foods, and Albertsons
2. `frame_retriever` - Retrieve the most relevant frames within the localized chart clip that clearly show the store names and the two rating categories needed for comparison.
   Query: customer survey chart or table showing Walmart, Target, Whole Foods, and Albertsons with Store Cleanliness and Value for Dollar percentages
3. `generic_purpose` - Read the grounded frames and extract the Store Cleanliness and Value for Dollar percentages for Walmart, Target, Whole Foods, and Albertsons, then compute which store has the largest discrepancy and the approximate difference in percentage points.
   Query: From these frames, identify the percentages for Store Cleanliness and Value for Dollar for Walmart, Target, Whole Foods, and Albertsons. Compute the absolute difference between those two percentages for each of the four stores, and report which store has the largest discrepancy and the approximate magnitude in percentage points. Quote the relevant values used.

## Files

- [plan json](planner/round_01_plan.json)
- [planner raw](planner/round_01_raw.txt)
