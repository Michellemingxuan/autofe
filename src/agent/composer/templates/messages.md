The user turns of a run - what the agent is told, after its brief, to start and
to move it on. Each `## name` below is one message; `{fields}` are filled in.

## direction
Direction: {direction}

## linkage
Write and propose the linkage for the source '{source}'.

## ideas
Stage 1 of 2 - ideas. Look at the data the direction needs (the tools are read-only here), then answer with your ideas - see Ideas first in your brief. Nothing is proposed in this stage.

## propose
Stage 2 of 2 - proposals. Your ideas are recorded: {names}. Spend your intents on them - the most promising and most different first, each under its idea's name (screen_feature for an L1/L2 idea, screen_request for an L3 one). Call report_findings only once the intents are spent, or nothing left is worth one.

## sent_back
Sent back: {error}. Fix it and answer again.

## nudge
You stopped without calling report_findings, so the run is not done. Carry on with the next step - call the tools you need - and finish by calling report_findings. Do not stop with a message alone.
