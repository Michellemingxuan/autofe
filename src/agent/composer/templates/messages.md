The user turns of a run - what the agent is told, after its brief, to start and
to move it on. Each `## name` below is one message; `{fields}` are filled in.

## direction
Direction: {direction}

## explore
No direction was given: this run is an open exploration of the task. Each round has a theme. Your first call is draw_theme - ideas given before it are sent back. Then look at the data the theme needs, and let the round's ideas follow it. The next round draws a new theme - a walk across the task, not one idea circled.

## linkage
Write and propose the linkage for the source '{source}'.

## ideas
Stage 1 of 2 - ideas. This round leans on the engines **{focus}**: at least {min_focus} ideas come from them - call `engine` for their methods. Look at the data the direction needs (the tools are read-only here), then answer with your ideas - see Ideas first in your brief, and the ideas skill. Nothing is proposed in this stage.

## propose
Round {round} - proposals. {budget}. Still wanted: {targets_left}.

### This round's ideas, in the order to work them
Drawn by Thompson sampling from each idea's record: a fresh idea and a strong one come early; one whose features carried no signal comes later - never left out. Each screen reply gives a fresh order (`work_next`).

{round_ideas}

### Each attempt
1. Take the first idea in the order with a measure not yet tried. Say in one sentence which idea and which measure you are trying.
2. Unsure of a column, a date format or how sparse the events are? Look first (run_probe) - a failed script costs an attempt, a probe does not.
3. Propose it, naming the idea it tests:
   * L1 / L2: `screen_feature(name, description, level, code, idea="<idea>")`. Name it for what it computes - `<measure>_<window>`, not the idea's name. If the reply says it reads like an earlier feature, resend with `differs="<what this measures that it does not>"`.
   * L3: `screen_request(gap, sql, source_name, features, idea="<idea>")`, or `propose_new_data(...)` beyond every scope - then `challenge_request`.
4. Read the reply before the next attempt:
   * refused (C0: same code, same meaning, same values; a name taken) - nothing was spent: change what it says, or move on;
   * explained by the model (C1, `nearest` names the column) - the model has it: the next measure must use other data, not another window of this one;
   * screened - the Gini gain against your gates ({gates}). `residual_rho` is information only: judge by the gain.
5. A measure that failed twice the same way is done. An idea is not: try its next measure, or follow `work_next`.

### Stop
Call report_findings when the target is reached, the attempts are used, or nothing left in the order is worth an attempt - say which, and why.

## sent_back
Sent back: {error}. Fix it and answer again.

## nudge
You stopped without calling report_findings, so the run is not done. Carry on with the next step - call the tools you need - and finish by calling report_findings. Do not stop with a message alone.

## next_round
Round {round} of ideas. So far: {budget}.
Worked: {worked}.
Failed: {failed}.
This round leans on the engines **{focus}**: at least {min_focus} ideas come from them.
Give {n} to {most} new ideas - none of the earlier ones - see Ideas first in your brief. Build on what worked; leave what failed, unless the reason says what would fix it.
