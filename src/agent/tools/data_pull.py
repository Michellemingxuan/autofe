"""L3: ask for data nobody has, as a rationale plus BigQuery SQL over the CAS scope.

Two ways a run uses it:

* **Alongside features** (L3 with L1/L2) - each request waits for the analyst's
  approval, as before; an approved one is run in BigQuery and comes back as a
  source.
* **A data-request run** (L3 only) - nothing is screened and nothing waits.
  Each proposal is recorded - rationale, the features it would enable, the SQL,
  the CAS tables it reads - and spends one intent. The agent then challenges it
  itself: can it be built from the current data? (``agent.tools.challenge``)
  It is kept or dropped; the run's summary carries the kept requests' SQL, and
  the analyst reviews kept and dropped in ``data_requests.md``.

Either way the SQL is read first, against the CAS column lists the analyst
provided, and sent back to the agent to fix - before it costs anything - if:

* it reads a table that is not in the CAS scope;
* it uses a column that table does not have (an invented ``customer_id`` or
  ``as_of_date`` is the usual slip);
* it selects none of the table's customer / account / card identifiers, so the
  result could not be linked to the model rows;
* it does not filter on the table's partition date - these tables are huge.
"""

from __future__ import annotations

import json
import re
from typing import Any

from agent.memory import same_request
from agent.session import Session
from agent.tools.ideas import ideas_needed

__all__ = ["screen_request", "sql_tables", "sql_columns", "save_requests",
           "write_requests_report"]


def sql_tables(sql: str) -> list[str]:
    """The tables a query reads - FROM and JOIN targets, minus its own CTEs."""
    text = re.sub(r"--[^\n]*|/\*.*?\*/", " ", sql, flags=re.S)
    ctes = {m.lower() for m in re.findall(r"\b(\w+)\s+as\s*\(", text, flags=re.I)}
    found = re.findall(r"\b(?:from|join)\s+([`\"\w.\-]+)", text, flags=re.I)
    tables = []
    for raw in found:
        name = raw.strip('`"')
        if name.lower() in ctes or name.lower() in ("unnest", "lateral"):
            continue
        tables.append(name)
    return list(dict.fromkeys(tables))


_SQL_WORDS = set("""
select from where and or not in is null as on join left right inner outer full cross group by
order having limit distinct case when then else end between like with union all except
intersect interval date datetime timestamp time day week month year quarter hour minute second
true false asc desc over partition rows range preceding following current row unnest qualify
using window exists any some offset nulls first last if count sum avg min max cast safe_cast
struct array extract isoweek dayofweek dayofyear
""".split())


def sql_columns(sql: str) -> set[str]:
    """The column names a query uses - minus keywords, functions, aliases and tables."""
    text = re.sub(r"--[^\n]*|/\*.*?\*/", " ", sql, flags=re.S)
    text = re.sub(r"'(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\"", " ", text)    # string literals
    text = re.sub(r"@\w+", " ", text)                                           # parameters
    tables = {t.lower() for t in sql_tables(sql)}
    aliases = {a.lower() for a in re.findall(r"\bas\s+`?(\w+)`?", text, flags=re.I)}
    aliases |= {a.lower() for a in re.findall(r"\)\s+(\w+)\s*(?=,|\bfrom\b|$)", text, flags=re.I)}
    aliases |= {c.lower() for c in re.findall(r"\b(\w+)\s+as\s*\(", text, flags=re.I)}   # CTEs
    aliases |= {a.lower() for a in re.findall(
        r"\b(?:from|join)\s+[`\w.\-]+`?\s+(?!on\b|where\b|join\b|left\b|inner\b|group\b|order\b)(\w+)",
        text, flags=re.I)}
    found = set()
    for match in re.finditer(r"`?([A-Za-z_][\w]*(?:\.[A-Za-z_][\w]*)*)`?(\s*\()?", text):
        name, call = match.group(1), match.group(2)
        if call or name.lower() in tables:
            continue
        last = name.split(".")[-1].lower()
        if ".".join(name.split(".")).lower() in tables or last in _SQL_WORDS or last in aliases:
            continue
        if any(name.lower().endswith(t.split(".")[-1]) for t in tables if "." in name):
            continue                                      # a qualified table name
        found.add(last)
    return found


def _check_columns(session: Session, sql: str, tables: list[str]) -> str | None:
    """Columns, identifier and partition filter, against the provided CAS columns."""
    profiles = [session.ws.table_profile(t.split(".")[-1]) for t in tables]
    known = {c.lower() for p in profiles for c in p["columns"]}
    if not known:
        return None
    unknown = sorted(sql_columns(sql) - known)
    if unknown:
        return (f"columns not in {[p['table'] for p in profiles]}: {unknown}. Use only the "
                "provided CAS columns - scope(table=...) lists them, with the identifiers "
                "and the partition date.")
    used = sql_columns(sql)
    identifiers = {c.lower() for p in profiles for c in p["identifiers"]}
    if identifiers and not used & identifiers:
        shown = [c for p in profiles for c in p["identifiers"]][:8]
        return (f"select a customer / account / card identifier so the result can be linked "
                f"to the model rows - e.g. {shown}")
    text = re.sub(r"--[^\n]*", " ", sql)
    where = text[re.search(r"\bwhere\b", text, re.I).start():] if re.search(r"\bwhere\b", text, re.I) else ""
    for p in profiles:
        if p["partition"] and not any(re.search(rf"\b{re.escape(c)}\b", where, re.I) for c in p["partition"]):
            return (f"filter {p['table']} on its partition date {p['partition']} in WHERE - "
                    "the CAS tables are very large; limit it to the model sample's date range")
    return None


def _unknown(session: Session, tables: list[str]) -> list[str]:
    scope = session.ws.scope()
    if not len(scope) or "table" not in scope:
        return []                                   # no scope to check against
    known = {str(t).lower() for t in scope["table"].dropna().unique()}
    return [t for t in tables if t.split(".")[-1].lower() not in known]


def screen_request(session: Session, gap: str, sql: str, source_name: str,
                      features: str = "") -> dict[str, Any]:
    """Propose a data pull: why, what it enables, and the SQL."""
    if "L3" not in session.params.levels:
        return {"approved": False, "error": "L3 data pulls are off for this run"}
    if session.intents_used >= session.K:
        return {"ok": False, "error": f"all {session.K} intents are used; call report_findings"}
    if (problem := session.level_full("L3")):
        return {"ok": False, "error": problem}
    if (missing := ideas_needed(session)):
        return missing
    if not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", source_name or ""):
        return {"ok": False, "error": f"{source_name!r}: name the source in snake_case"}
    tables = sql_tables(sql)
    if not tables:
        return {"ok": False, "error": "the SQL selects from no table"}
    unknown = _unknown(session, tables)
    if unknown:
        known = sorted(session.ws.scope()["table"].dropna().unique().tolist())
        return {"ok": False, "error": f"not in the CAS scope: {unknown}; the scope's tables "
                                      f"are {known[:20]}. Fix the SQL - nothing was spent."}

    if (problem := _check_columns(session, sql, tables)):
        return {"ok": False, "error": problem + " Nothing was spent."}

    if (twin := same_request(session, tables, sorted(sql_columns(sql)))):
        return {"ok": False, "error": f"this asks for {twin}. Nothing was spent - ask for "
                                      "different data, or build on what is already requested."}

    if session.l3_only:
        return _record(session, gap, sql, source_name, features, tables)

    # A mixed run: the request spends one of the run's L3 intents, then waits
    # for the analyst.
    session.intents_used += 1
    payload = {"intent": f"R{len(session.data_requests) + 1}", "gap": gap, "sql": sql,
               "source_name": source_name, "features": features, "tables": tables,
               "columns": sorted(sql_columns(sql))}
    decision = session.ask("data_pull", payload)
    session.data_requests.append({**payload, "approved": decision.approved, "spent": True,
                                  "note": decision.note})
    save_requests(session)
    if not decision.approved:
        return {"approved": False, "user_note": decision.note}
    return {"approved": True, "user_note": decision.note,
            "next": (f"the analyst will run it in BigQuery and drop {source_name}.parquet "
                     f"+ {source_name}_data_sample.json into the additional data "
                     "folder; it appears in catalog() when it lands. Carry on with "
                     "other intents meanwhile.")}


def _record(session: Session, gap: str, sql: str, source_name: str, features: str,
            tables: list[str]) -> dict[str, Any]:
    """A data-request run: keep the proposal for review; spend one intent."""
    if any(r["source_name"] == source_name for r in session.data_requests):
        return {"ok": False, "error": f"{source_name!r} is already proposed; choose a new name"}
    session.intents_used += 1
    # Numbered by proposal, not by intent: a dropped request is refunded, and
    # the next one must not reuse its number.
    record = {"intent": f"R{len(session.data_requests) + 1}", "spent": True,
              "source_name": source_name, "gap": gap,
              "features": features, "sql": sql.strip(), "tables": tables,
              "columns": sorted(sql_columns(sql)), "status": "proposed"}
    session.data_requests.append(record)
    save_requests(session)
    session.emit("data_request", **record, intents_used=session.intents_used, K=session.K)
    reply: dict[str, Any] = {"recorded": True, "intent": record["intent"], "tables": tables,
                             "budget": session.budget()}
    reply["next"] = (f"challenge {record['intent']}: can the current data supply it? "
                     "Call challenge_request with your verdict.")
    return reply


def save_requests(session: Session) -> None:
    (session.run_dir / "data_requests.json").write_text(
        json.dumps(session.data_requests, indent=2, default=str))
    folder = session.run_dir / "data_requests"
    folder.mkdir(exist_ok=True)
    for r in session.data_requests:
        name = (f"{r['intent']}_" if r.get("intent") else "") + f"{r['source_name']}.sql"
        if r.get("status") == "dropped":
            (folder / name).unlink(missing_ok=True)          # only kept requests get a file
            continue
        (folder / name).write_text(f"-- {r['gap'].strip()}\n{r['sql'].strip()}\n")
    write_requests_report(session)


def write_requests_report(session: Session) -> str:
    """Every request as one markdown document - kept first, then dropped and why."""
    kept = [r for r in session.data_requests if r.get("status") != "dropped"]
    dropped = [r for r in session.data_requests if r.get("status") == "dropped"]
    lines = [f"# Data requests - {session.direction}", "",
             f"Run `{session.run_id}` · {len(kept)} kept, {len(dropped)} dropped by the "
             "challenge (can be built from current data)", ""]

    def challenge_lines(r: dict[str, Any]) -> list[str]:
        c = r.get("challenge")
        if not c:
            return []
        out = [f"**Challenge - can it be built from current data?** {c['verdict']}", "",
               c.get("reasoning", ""), ""]
        if c.get("note"):
            out += [f"_{c['note']}_", ""]
        if c.get("code"):
            state = "ran on the screen rows" if c.get("code_ok") else "did not run"
            out += [f"Construction from current data ({state}):", "", "```python",
                    c["code"], "```", ""]
        return out

    for title, group in (("Kept", kept), ("Dropped", dropped)):
        if not group:
            continue
        lines += [f"# {title}", ""]
        for r in group:
            lines += [f"## {r.get('intent', '')} `{r['source_name']}`", "",
                      "**Why it is needed**", "", r["gap"].strip(), ""]
            if r.get("features"):
                lines += ["**Features it would enable**", "", str(r["features"]).strip(), ""]
            lines += [f"**Reads** {', '.join(f'`{t}`' for t in r.get('tables', []))}"
                      + (f" - columns {', '.join(f'`{c}`' for c in r['columns'])}"
                         if r.get("columns") else ""), ""]
            lines += challenge_lines(r)
            lines += ["```sql", r["sql"].strip(), "```", ""]
    text = "\n".join(lines)
    (session.run_dir / "data_requests.md").write_text(text)
    return text
