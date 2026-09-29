# Agent evaluation

`scripts/alfworld_agentbench.py` implements Writes Off, Closed Loop, Verified
Writes, failure-only writes, and Settlement with persistent LoRA state. It can
run frozen-block evaluation or online/prequential evaluation. The latter is the
appropriate test when TTT is expected to remain enabled during deployment.

The official AgentBench ReAct prompt and action parser are external assets; pass
an explicit prompt path and record the AgentBench revision. ALFWorld task types
are: 1 Pick & Place, 2 Examine in Light, 3 Clean & Place, 4 Heat & Place, 5 Cool
& Place, and 6 Pick Two & Place.

ALFWorld is environment-grounded and episodes are capped, so it does not mimic
a 128K autonomous language stream. Interpret a null Closed–Writes Off gap as a
boundary result for this benchmark, not proof that persistent self-writing is
always harmless.

## WebShop and ScienceWorld

`ws_arms.py` and `sw_arms.py` retain the paper-era policy implementations.
They require the upstream WebShop checkout and ScienceWorld installation,
respectively. Run the corresponding probe/canary scripts before a full matrix;
a task with a zero frozen success rate cannot establish the value of selective
write admission. Record the upstream repository revision in each result.
