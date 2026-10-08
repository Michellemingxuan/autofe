The user turns of a run - what the agent is told, after its brief, to start and
to move it on. Each `## name` below is one message; `{fields}` are filled in.

## direction
Direction: {direction}

## explore
No direction was given: this run is an open exploration of the task. Each round has a theme. Your first call is draw_theme - ideas given before it are sent back. Then look at the data the theme needs, and let the round's ideas follow it. The next round draws a new theme - a walk across the task, not one idea circled.

## linkage
Write and propose the linkage for the source '{source}'.

## ideas
Stage 1 of 2 - ideas. Look at the data the direction needs (the tools are read-only here), then answer with your ideas - see Ideas first in your brief. Nothing is proposed in this stage.

## propose
Stage 2 of 2 - proposals. Your ideas are recorded: {names}. Work toward your target with them - the most promising and most different first, each under its idea's name (screen_feature for an L1/L2 idea; screen_request or propose_new_data for an L3 one). Call report_findings once the target is reached or the attempts are used - or when nothing left is worth an attempt.

## sent_back
Sent back: {error}. Fix it and answer again.

## nudge
You stopped without calling report_findings, so the run is not done. Carry on with the next step - call the tools you need - and finish by calling report_findings. Do not stop with a message alone.

## next_round
Round {round} of ideas. So far: {budget}.
Worked: {worked}.
Failed: {failed}.
Give {n} to {most} new ideas - none of the earlier ones - see Ideas first in your brief. Build on what worked; leave what failed, unless the reason says what would fix it.
