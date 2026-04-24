# Round 02 Plan

## Strategy

Repair the unsupported identity-to-option mapping by grounding which visible person is Mary during the vote/proposal scene, then use that visual evidence together with the already-supported transcript facts to decide whether the question targets Mary or the children.

## Refinement Instructions

Preserve the already-supported transcript facts: someone says 'she pitched the water slide idea,' the vote prompt is 'raise your hand if you want to do her idea,' the speaker notes 'It's all the kids,' and then says 'Mary, you won the vote.' Replace the unsupported claim that Mary is 'the person in red' unless step 3 directly grounds that mapping from the vote-scene frames. Use the new visual evidence only to resolve the identity-to-option mapping and the interpretation of the question target. If step 3 confirms that Mary is the woman in red and that the question is asking for the named person advocating the idea rather than every supporter, update the trace toward option B. If the new evidence instead shows only that the children want it while Mary-to-red remains ungrounded, do not keep option B by inference; prefer the directly grounded group support or state residual ambiguity. Do not erase the fact that the children also raised their hands unless the new evidence directly contradicts it.

## Steps

1. `visual_temporal_grounder` - Localize the vote scene where raised hands identify supporters of the water-slide idea and where Mary is addressed by name.
   Query: group on the water park stage voting by raising hands about pushing the body down the slide, including the woman in a red hoodie raising her hand
2. `frame_retriever` - Retrieve frames from the localized vote clip that clearly show which people have their hands raised, especially the woman in red and the children.
   Query: water park stage vote scene showing raised hands for pushing the body down the slide, with the woman in a red hoodie and the children visible
3. `generic_purpose` - Interpret the retrieved vote-scene frames to determine whether the named winner Mary is the woman in red and to distinguish proposer/winner from the broader group of children who also want the slide plan.
   Query: Using these vote-scene frames and the transcript facts that the speaker says 'raise your hand if you want to do her idea' and later says 'It's all the kids' and 'Mary, you won the vote,' identify which visible person Mary refers to among the answer choices and explain whether the question 'Who wants to send the body into the water slide?' is best matched by the named proposer-winner or by the children as a group.

## Files

- [plan json](planner/round_02_plan.json)
- [planner raw](planner/round_02_raw.txt)
