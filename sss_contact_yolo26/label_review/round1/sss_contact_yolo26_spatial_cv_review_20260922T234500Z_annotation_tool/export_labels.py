#!/usr/bin/env python3
"""Export working copy A to a new versioned folder + ZIP (same code as the UI button).

    python3 export_labels.py [--note "text"] [--purpose final_labels|checkpoint|acceptance_test]

Prefer the web UI button while the server is running; this command only reads A.
"""
import argparse
import json
import os
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import serve  # noqa: E402
from sss_tool.common import DEFAULT_B_DIR, DEFAULT_ZIP  # noqa: E402
from sss_tool.exporter import export_a  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--note", default="")
    ap.add_argument("--purpose", default="final_labels")
    ap.add_argument("--zip", default=DEFAULT_ZIP)
    ap.add_argument("--backup-b", default=DEFAULT_B_DIR)
    a = ap.parse_args()
    ctx = serve.startup(os.path.abspath(a.zip), os.path.abspath(a.backup_b))
    r = export_a(ctx["store"], ctx["index"], ctx["zip_path"], ctx["b_dir"], ctx["b_manifest_path"],
                 note=a.note, purpose=a.purpose)
    print(json.dumps(r, indent=1))
    sys.exit(0 if r["verification"]["ok"] else 1)


if __name__ == "__main__":
    main()
