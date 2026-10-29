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
Already in this run's draw, from earlier runs, with their records: {loaded}. Give new ideas, not these again; to bring in another earlier idea, name it in `carry`.

## propose
Round {round} - proposals. {budget}. Still wanted: {targets_left}.

### The ideas, in the order to work them
This round's new ideas, the earlier rounds' and the earlier runs' ideas carried in (`from run`) together - each with its record from every run. Drawn by Thompson sampling from each idea's record: a fresh idea and a strong one come early - an earlier round's that carried signal too; one whose features carried no signal comes later - never left out. Each screen reply gives a fresh order (`work_next`).

{round_ideas}

### Each attempt
1. Work the idea at the top of the order - the latest `work_next`, a fresh draw after every screen. Ideas are drawn in proportion to their chance of being useful: a strong one comes up often, one whose features missed comes up less often, but still comes up. An idea is not used up by its listed measures: when they are tried, derive a new measure of the same hypothesis (`measures_left` says how many listed ones remain). Say in one sentence which idea and which measure you are trying; skip the top idea only with a reason, and say it.
2. Unsure of a column, a date format or how sparse the events are? Look first (run_probe) - a failed script costs an attempt, a probe does not.
3. Propose it, naming the idea it tests:
   * L2 on a long source: summarise it first - `define_aggregate(name, source, period, description, code, window_days, key)`, the temporary features per transaction (`period="transaction"`, the events in the `window_days` before each as-of date) or per customer and day, week or month, grouped by its `key` (no attempt spent) - unless an aggregate of this run already has what the measure needs. Build the feature on `sources["<aggregate>"]`.
   * L1 (model columns only) / L2 (reads a source or an aggregate): `screen_feature(name, description, level, code, idea="<idea>")`. Name it for what it computes - `<measure>_<window>`, not the idea's name. If the reply says it reads like an earlier feature, resend with `differs="<what this measures that it does not>"`. Write whole-frame code: no loops over rows, ids or groups (`iterrows`, `apply(axis=1)`, `for ... in df.groupby(...)`) - they are refused before they run; the feature skill's "No loops" table has the replacement for each.
   * L3: `screen_request(gap, sql, source_name, features, idea="<idea>")`, or `propose_new_data(...)` beyond every scope - then `challenge_request`.
4. Read the reply before the next attempt:
   * refused (C0: same code, same meaning, same values; a name taken) - nothing was spent: change what it says, or move on;
   * explained by the model (C1, `nearest` names the column) - the model has it: the next measure must use other data, not another window of this one;
   * screened - the Gini gain against your gates ({gates}). `variable_gini` (the feature alone) is information only: judge by the gain.
5. A feature that fails tests one measure, not the idea: the idea stays in the draw, a little less likely to come up. When it comes up again, try a different measure - another data source, statistic or comparison, not another window of the same one. A measure that failed twice the same way is done. An idea is done only when the reply says it cannot work: the model already has it (C1 names the column), or the data it needs is empty.

### Stop
Call report_findings when the target is reached, the attempts are used, or nothing left in the order is worth an attempt - say which, and why.

## sent_back
Sent back: {error}. Fix it and answer again.

## nudge
You stopped without calling report_findings, so the run is not done. Carry on with the next step - call the tools you need - and finish by calling report_findings. Do not stop with a message alone.

## rethink
Not yet. Before you call this direction exhausted, take stock of where it stands.
So far: {budget}.
Verified: {worked}.
Why the other attempts failed: {failed}. The latest:
{examples}
This is a quite challenging task. Think out of the box: analyse why the earlier trials failed, and how each could be changed to lift the performance - a different signal rather than another window of the same one, a source or interaction not tried yet, a transformation of what the model already explains, the opposite of an assumption the failures share. Write that analysis in a few lines; a new round of ideas built on it comes next.

## next_round
Round {round} of ideas. So far: {budget}.
Worked: {worked}.
Failed: {failed}.
This round leans on the engines **{focus}**: at least {min_focus} ideas come from them.
Give {n} to {most} new ideas - none of the earlier ones - see Ideas first in your brief. Build on what worked; leave what failed, unless the reason says what would fix it. The earlier ideas stay in the draw: each attempt goes to whichever idea the draw puts first, new or earlier.
