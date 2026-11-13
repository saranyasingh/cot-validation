"""
Patch a CharmChat checkout's ToolUniverse MCP server so the CureBench verifier agent gets whole
drug-label sections. Safe to run repeatedly; run_curebench_charmchat.slurm runs it before starting
the backend, followed by `uv lock` + `uv sync` so the version bump takes effect.

1. Require tooluniverse>=1.5.9. CharmChat's lock pins 1.1.4, whose FDA_get_drug_label silently cuts
   every label section to 2,000 characters (1.5.9: 25,000 per section, and it reports any cut).
2. Format lists of records as text when they carry long fields. The wrapper renders every list of
   dicts as a markdown table with cells cut to 80 characters, so the per-section FDA tools
   (adverse reactions, patient information, ...) reached the agent as one line of each section.
3. Add the per-section FDA label tools to the curated tool list, so the agent can fetch just the
   section it needs instead of whole labels.

Usage: python slurm_scripts/patch_charmchat.py ~/charmchat
"""

import re
import sys
from pathlib import Path

MARKER = "cot-validation patch"
MIN_TOOLUNIVERSE = "1.5.9"

FDA_SECTION_TOOLS = [
    "FDA_get_indications_by_drug_name",
    "FDA_get_adverse_reactions_by_drug_name",
    "FDA_get_boxed_warning_info_by_drug_name",
    "FDA_get_warnings_and_cautions_by_drug_name",
    "FDA_get_warnings_by_drug_name",
    "FDA_get_precautions_by_drug_name",
    "FDA_get_do_not_use_info_by_drug_name",
    "FDA_get_stop_use_info_by_drug_name",
    "FDA_get_info_for_patients_by_drug_name",
    "FDA_get_patient_package_insert_from_drug_name",
    "FDA_get_instructions_for_use_by_drug_name",
    "FDA_get_info_for_nursing_mothers_by_drug_name",
    "FDA_get_pregnancy_or_breastfeeding_info_by_drug_name",
    "FDA_get_pediatric_use_info_by_drug_name",
    "FDA_get_geriatric_use_info_by_drug_name",
    "FDA_get_overdosage_info_by_drug_name",
    "FDA_get_mechanism_of_action_by_drug_name",
    "FDA_get_dosage_forms_and_strengths_by_drug_name",
]

FORMAT_TABLE = f'''def _format_table(rows: list[dict]) -> str:
    # {MARKER}: records with long text fields (drug-label sections) are written out in full
    # instead of as table cells cut to 80 characters.
    if not rows:
        return ""
    if any(isinstance(v, (dict, list)) or (isinstance(v, str) and len(v) > MAX_CELL_CHARS)
           for row in rows for v in row.values()):
        parts = []
        for i, row in enumerate(rows[:MAX_TEXT_RECORDS], 1):
            fields = {{k: (v[:MAX_FIELD_CHARS] + f" [... {{len(v) - MAX_FIELD_CHARS}} more characters cut]"
                          if isinstance(v, str) and len(v) > MAX_FIELD_CHARS else v)
                      for k, v in row.items()}}
            parts.append(f"### Record {{i}}\\n" + _format_dict(fields))
        if len(rows) > MAX_TEXT_RECORDS:
            parts.append(f"*... and {{len(rows) - MAX_TEXT_RECORDS}} more records*")
        return "\\n\\n".join(parts)
    keys = list(dict.fromkeys(k for row in rows for k in row))[:MAX_TABLE_COLUMNS]
    header = "| " + " | ".join(k.replace("_", " ").title() for k in keys) + " |"
    separator = "| " + " | ".join("---" for _ in keys) + " |"
    lines = [header, separator]
    for row in rows[:MAX_TABLE_ROWS]:
        cells = [str(row.get(k, "")).replace("|", "/").replace("\\n", " ")[:MAX_CELL_CHARS] for k in keys]
        lines.append("| " + " | ".join(cells) + " |")
    if len(rows) > MAX_TABLE_ROWS:
        lines.append(f"\\n*... and {{len(rows) - MAX_TABLE_ROWS}} more rows*")
    return "\\n".join(lines)


MAX_TABLE_ROWS = 250
MAX_TABLE_COLUMNS = 12
MAX_CELL_CHARS = 300
MAX_TEXT_RECORDS = 5
MAX_FIELD_CHARS = 20000


'''


def patch_pyproject(path: Path) -> str:
    text = path.read_text()
    required = f'"tooluniverse>={MIN_TOOLUNIVERSE}"'
    if required in text:
        return "pyproject.toml: already requires tooluniverse>=" + MIN_TOOLUNIVERSE
    new, n = re.subn(r'"tooluniverse\s*(?:[<>=!~;][^"]*)?"', required, text, count=1)
    if n != 1:
        raise SystemExit(f"pyproject.toml: no tooluniverse dependency found in {path}")
    path.write_text(new)
    return f"pyproject.toml: now requires tooluniverse>={MIN_TOOLUNIVERSE}"


def patch_formatting(path: Path) -> str:
    text = path.read_text()
    if MARKER in text:
        return "formatting.py: already patched"
    new, n = re.subn(r'def _format_table\(rows: list\[dict\]\) -> str:.*?(?=\ndef format_json)',
                     lambda _: FORMAT_TABLE, text, flags=re.DOTALL)
    if n != 1:
        raise SystemExit(f"formatting.py: could not find _format_table in {path}")
    path.write_text(new)
    return "formatting.py: long-text records are no longer cut to 80-character table cells"


def patch_config(path: Path) -> str:
    text = path.read_text()
    if MARKER in text:
        return "config.py: already patched"
    tools = "".join(f'    "{t}",\n' for t in FDA_SECTION_TOOLS)
    text += (f"\n\n# --- {MARKER}: per-section FDA label tools for the CureBench verifier ---\n"
             f"if CURATED_TOOLS is not None:\n    CURATED_TOOLS |= {{\n{tools}    }}\n")
    path.write_text(text)
    return f"config.py: added {len(FDA_SECTION_TOOLS)} per-section FDA label tools"


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    server = Path(sys.argv[1]).expanduser() / "mcp-servers" / "tooluniverse-mcp"
    package = server / "src" / "tooluniverse_mcp"
    for message in (patch_pyproject(server / "pyproject.toml"),
                    patch_formatting(package / "formatting.py"),
                    patch_config(package / "config.py")):
        print(f"[patch_charmchat] {message}")
