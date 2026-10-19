# Modeling Agent: demo video script

**Audience:** senior leaders. **Length:** about 6½ minutes. **Format:** a screen recording of the AutoFE workspace, cut together with the architecture figures, plus a voiceover. Everything is edited after recording.

**The one line to leave them with:**
> An analyst gives the agent a direction. The agent proposes ideas, builds and checks the features, and proves them on data it never saw, down to the cases the model would have caught sooner.

**Three things the video must show:**
1. **It starts from the analyst's intuition.** The agent doesn't run on its own; the analyst sets the direction.
2. **It's rigorous.** Every feature passes checks, and every number comes from held-out data.
3. **It learns in loops.** Results feed back into the ideas, the data and the analyst's next direction.

---

## Before recording

**Prepare the workspace**
- [ ] Use the real workspace (`cdss_us_sbs`, in the private environment). Rehearse first on `synthetic_agent`.
- [ ] Confirm the sources are linked and the target cases (e.g. high-balance defaulters) are loaded on Setup.
- [ ] **Pre-run the directions you'll show.** Runs take far longer than the video allows. Record one live run for its first minute, and use a finished run for everything after.
  - Run 1, with L1 and L2: *"Use payment behaviour relative to spending to find risk the bureau columns miss."*
  - Run 2, with L3: *"Are there CAS authorization variables the model does not use that signal distress?"* This run produces data requests.
- [ ] Finish at least one evaluation with single features and one combination, so Metrics results and Inference results have content.
- [ ] Write down the real numbers to quote (test Δ Gini, number of verified features, cases flagged sooner). Fill them into the `[ ]` placeholders below. **Don't quote numbers that aren't on screen.**

**Prepare the screen**
- [ ] Record at 1920×1080 with the browser in full screen and zoom at 110–125%, so text reads on a projector.
- [ ] Hide bookmarks, notifications and other tabs. Check there's no customer-level data on screen that shouldn't be shown; if there is, blur it in the edit.
- [ ] Clear session storage so the opening animation plays (or click the logo to replay it).
- [ ] Export the figures as PNGs at 2× for the cut-ins:
  - `modeling_agent_architecture_abstract_v2.svg`
  - `modeling_agent_architecture_loop.svg`

**Recording tips**
- Record each scene as its own clip, which makes the edit easier.
- Move the mouse slowly, and pause 2 seconds on each screen before and after any action.
- Record the voiceover separately, after the cut is locked.

---

## Scene by scene

Times are targets for the final cut.

### Scene 0 · Opening (0:00–0:20)

| On screen | Voiceover |
|---|---|
| The AutoFE opening animation: data layers, the scan, and features lighting up. The title reads "Automated feature discovery and model enhancement." Click **Enter the workspace**, and the dive transition plays. | "Improving a credit model usually means months of analysts hand-crafting features. We built an agent that does the heavy lifting, while the analyst stays in charge of the direction." |

**Edit:** keep the full animation at real speed. Start the music softly here.

### Scene 1 · The idea in one picture (0:20–0:55)

| On screen | Voiceover |
|---|---|
| Cut to the abstract_v2 figure. Slowly zoom from left to right: inputs, then the agent's three stages (Discover, Evaluate, Infer), then the output. | "It works in three stages. **Discover**: from the analyst's direction, the agent proposes ideas and turns them into features, each one quickly verified. **Evaluate**: the strongest features are tested on data the agent never saw. **Infer**: we score real target cases to see who the new model would have caught sooner." |

**Edit:** highlight each stage card as it's named (a soft glow or dim the rest). Don't explain the loops yet; they come in Scene 7.

### Scene 2 · Setup (0:55–1:25)

| On screen | Voiceover |
|---|---|
| The Setup page. Pan over the base model's card, the data-pull scope and additional sources (showing their **linked** badges), and the target cases. | "The analyst points the agent at what the model already has, the wider warehouse it may pull from, and the cases we care most about, such as high-balance defaulters. New data is only used once its linkage is confirmed." |

**Edit:** add a callout on one **linked** badge and one on the target cases. Speed up scrolling 2×.

### Scene 3 · Give it a direction (1:25–2:00)

| On screen | Voiceover |
|---|---|
| Click Discover, then **+ New direction**. Type Run 1's direction (speed up the typing in the edit). Hover over the levels L1, L2 and L3, then click Start. | "The analyst writes a direction in plain words. The levels set how far the agent may go: combine what the model already has, bring in a new source, or ask for data the model doesn't have yet." |

**Edit:** zoom in on the direction box while typing. Add a one-line caption under each level as the cursor reaches it.

### Scene 4 · The agent at work (2:00–3:15)

| On screen | Voiceover |
|---|---|
| **(a)** The live run page, with the agent trace on the left and the timeline on the right. Ideas appear. | "The agent proposes ideas: each one a mechanism it thinks drives risk." |
| **(b)** Jump cut to the finished Run 1. Scroll the trace: features being built, with check results C0–C3. | "Every idea becomes code, and every feature goes through checks. Have we seen it before? Is it redundant with what the model already has? Does it carry signal on its own? Does it add to the model? Most ideas fail here, and that's the point." |
| **(c)** The idea list re-ordering after a round. | "Ideas that carry signal move up and get more attempts. The weak ones aren't thrown away; they're tried later." |
| **(d)** Run 2 (L3): a data request with its SQL and the data it needs, then click Approve. | "When the signal lives in data the model doesn't use, the agent writes the request itself, with the SQL and a reason. The analyst approves it, and the pulled data becomes a new source." |

**Edit:** this is the longest scene; keep it moving.
- Speed up (a) 4–8× with a small "time-lapse" tag.
- In (b), freeze-frame one feature row and annotate its four checks with ✓/✗.
- In (d), circle the **Approve** button.
- Optionally open **Model calls** for 3 seconds to show the agent's reasoning is fully logged.

### Scene 5 · The logging board (3:15–3:40)

| On screen | Voiceover |
|---|---|
| Open the **Logging board**. Filter to one idea and show the features and requests built from it, their checks and code, and whether each reached evaluation. | "Nothing is a black box. Every idea, every feature, every check and every line of code is on the board, along with why each one was kept or dropped." |

**Edit:** a slow push-in on the board. Add a caption: "Full audit trail."

### Scene 6 · Evaluate (3:40–4:35)

| On screen | Voiceover |
|---|---|
| **(a)** Step 3 · Evaluate, then the **Feature set** page. Tick a few verified features and one combination. | "The analyst picks the most promising features, alone or in combination." |
| **(b)** **Metrics results**: the verdict (PASS/FAIL), test Δ Gini, and the SHAP rank in the combined model. | "They're judged on the held-out test set, against the current model. Here, adding [features] lifts test Gini by [Δ], and the new features rank [#] in importance." |

**Edit:** put a big on-screen number for the Δ Gini, but only the real one from the screen.

### Scene 7 · Inference: the cases (4:35–5:25)

| On screen | Voiceover |
|---|---|
| **Inference results**. Pick the evaluated model and show the target cases split into sooner / same row / later / neither flags. Open one "sooner" case's timeline: the base and new scores over time, with the flag points. | "A Gini number is abstract, so we look at real cases. Of our [N] high-balance defaulters, the new model flags [n] of them sooner, a median of [k] rows ahead of today's model. In this case, it raises the flag [x] months before the current model does." |

**Edit:** this is the emotional peak. Slow down, zoom into the gap between the two flag points and label it "[k] earlier." Let the music lift slightly.

### Scene 8 · It learns in loops (5:25–6:05)

| On screen | Voiceover |
|---|---|
| Cut to the loop figure. Highlight the loops one at a time, A → D. | "What makes it an agent, not a pipeline, is the loops. **A**: quick check results re-rank the ideas. **B**: evaluation and case results sharpen what it generates next. **C**: data it pulls becomes a new source to build from. **D**: the findings go back to the analyst and inspire the next direction." |

**Edit:** animate each loop path with a glow as it's named (about 8 seconds each).

### Scene 9 · Close (6:05–6:30)

| On screen | Voiceover |
|---|---|
| The abstract_v2 figure, full frame, fading to a title card: "Modeling Agent · AutoFE", with a contact line and next steps. | "The analyst brings the intuition. The agent does the search, the checking and the proof. Next, we [ask: e.g. extend it to the next model / connect more warehouse sources / pilot with a model team]." |

**Edit:** hold the end card for 4 seconds, then fade out.

---

## Post-editing checklist

- **Captions:** burn in subtitles. Many leaders watch muted.
- **Pace:** jump-cut every wait, loading spinner and typing. No screen should stay still longer than 6 seconds without a zoom or callout.
- **Callouts:** use one visual style throughout, matching the figures: rounded labels on the purple #7A5AA8 or green #3F7D5A, with a soft drop shadow.
- **Time-lapse tags:** mark every sped-up section, so no one thinks runs take seconds.
- **Numbers:** every quoted number must appear on screen at that moment. If the data is illustrative, add a small footer saying so.
- **Privacy:** blur account IDs and any customer-level fields.
- **Length:** if it runs long, cut Scene 5 first, then 4(c). Never cut Scene 7.

## A 90-second cut (optional)

For a teaser or an opener, use Scene 0 (10s), Scene 1 (20s), Scenes 3 and 4(a) as a time-lapse (20s), the Scene 6 number (10s), the Scene 7 case (20s), and the Scene 9 end card (10s).

## Likely questions from leaders, with a one-line answer each

- **"Can it make things up?"** Every feature is code that runs on real data and must pass the checks and a held-out test; anything that fails is logged, not used.
- **"Who's in control?"** The analyst: they set the direction, confirm every new data linkage and approve every data request.
- **"How long does a run take, and what does it cost?"** [Typical run time and the token or model-call cost from the run's Model calls.]
- **"Is it ready for production?"** The features it proposes go through the same model governance as any analyst's features. The agent shortens discovery; it doesn't skip validation.
