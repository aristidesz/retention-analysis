"""Remove per-user outputs from the executed notebook so it can be committed.

Keeps charts and aggregate tables. Drops tables that show user rows (marked by
the "…" that `peek` adds to user ids), the single-user snapshot plots, and
anything that still contains a raw user id.

    python notebook/scrub_outputs.py notebook/retention_walkthrough.ipynb
"""

import json
import re
import sys

path = sys.argv[1]
nb = json.load(open(path))

for cell in nb["cells"]:
    if cell["cell_type"] != "code":
        continue
    if "show_snapshot(" in "".join(cell["source"]):
        cell["outputs"] = []  # one user's payments and features
        continue
    cell["outputs"] = [o for o in cell.get("outputs", []) if "…" not in json.dumps(o, ensure_ascii=False)]

left = re.findall(r"b['\"]?\\\\x", json.dumps(nb))
assert not left, f"raw user ids still in outputs: {len(left)}"
json.dump(nb, open(path, "w"), indent=1)
