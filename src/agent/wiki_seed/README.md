# Idea-engine wiki

What the idea engines know beyond the data. Layout, access and ingest:
`docs/wiki_layout.md` in autofe; the code: `src/agent/wiki.py`, `wiki_ingest.py`.

**Add material in `raw/` only** - markdown or text: notes, case reviews, studies.
The rest of this folder is kept by the agent: each raw file is folded into the
pages by small edits and committed (`git log` here shows every change and why;
`log.md` too). Edits by hand outside `raw/` are kept but recorded as such.

* `domains/<domain>/` - underwriting, customer_management, fraud, marketing: an
  `index.md`, pages true across the domain, and subdomains - one use case each
* `domains/<domain>/<subdomain>/` - `index.md`, subpages, and `cases/` (one page per reviewed
  case, named by its case id - `CR-20931.md`: Key points, Findings, Patterns - and
  `summary.md`: what they teach, without ids). The patterns are the themes open
  explorations draw.
* `cognitive/` - the idea engines, copied from the code
* `experiments/<model>/` - reflections on a model's results
