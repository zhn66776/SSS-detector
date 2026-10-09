#!/usr/bin/env python3
"""Read-only integrity check (writes nothing).

    python3 verify_integrity.py                     # ZIP + B unchanged? A vs B summary
    python3 verify_integrity.py --export A_final_labels_v001   # also re-verify one export

Checks:
  * source ZIP SHA-256 / size / mtime == values recorded when the platform was initialised
  * every B label file == backup_manifest.json SHA-256; no extra/missing files in B
  * B backup_manifest.json SHA-256 unchanged since initialisation
  * A has all 9000 label files, all parse as valid YOLO; which patches differ from B
  * (optional) an export's files, ZIP and manifest hashes agree
"""
import argparse
import hashlib
import json
import os
import sys
import zipfile

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from sss_tool.labels import parse_label_text, validate_boxes  # noqa: E402


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", help="export name to re-verify, e.g. A_final_labels_v001")
    a = ap.parse_args()
    man = json.load(open(os.path.join(HERE, "review_manifest.json")))
    init = man["source_fingerprints_at_initialisation"]
    ok = True
    zp = man["roles"]["ZIP"]["path"]
    st = os.stat(zp)
    zsha = sha(zp)
    zok = zsha == init["zip"]["sha256"] and st.st_mtime_ns == init["zip"]["mtime_ns"] and st.st_size == init["zip"]["size_bytes"]
    ok &= zok
    print("ZIP  %s\n     sha256 %s  size %d  mtime_ns %d  -> %s" % (zp, zsha, st.st_size, st.st_mtime_ns,
                                                                    "UNCHANGED" if zok else "CHANGED!"))
    bdir = man["roles"]["B"]["path"]
    bman_p = man["roles"]["B"]["backup_manifest"]
    bman_ok = sha(bman_p) == init["b_backup_manifest"]["sha256"]
    bman = json.load(open(bman_p))
    listed = {f["path"]: f for f in bman["files"]}
    bad = [p for p, f in listed.items() if sha(os.path.join(bdir, p)) != f["sha256"]]
    ldir = os.path.join(bdir, "validation/ground_truth_labels")
    present = {"validation/ground_truth_labels/" + n for n in os.listdir(ldir)}
    extra, missing = sorted(present - set(listed)), sorted(set(listed) - present)
    bok = bman_ok and not bad and not extra and not missing
    ok &= bok
    print("B    %s\n     manifest unchanged: %s; files checked %d; hash mismatches %d; extra %d; missing %d -> %s"
          % (bdir, bman_ok, len(listed), len(bad), len(extra), len(missing), "UNCHANGED" if bok else "PROBLEM!"))
    adir = man["roles"]["A"]["path"]
    diff, invalid, boxes = [], [], 0
    for p in listed:
        n = os.path.basename(p)
        ap_ = os.path.join(adir, n)
        if not os.path.exists(ap_):
            invalid.append(n + " (missing)")
            continue
        data = open(ap_, "rb").read()
        try:
            boxes += len(validate_boxes(parse_label_text(data.decode("ascii"))))
        except Exception as e:  # noqa: BLE001
            invalid.append("%s (%s)" % (n, e))
        if data != open(os.path.join(bdir, p), "rb").read():
            diff.append(n[:-4])
    ok &= not invalid
    print("A    %s\n     files %d; invalid %d; boxes %d; patches differing from B: %d %s"
          % (adir, len(listed), len(invalid), boxes, len(diff), diff[:20]))
    if a.export:
        ed = os.path.join(HERE, "exports")
        m = json.load(open(os.path.join(ed, a.export + "_manifest.json")))
        probs = []
        for f in m["files"]:
            p = os.path.join(ed, a.export, f["path"])
            if sha(p) != f["sha256"]:
                probs.append(f["path"])
        zpath = os.path.join(ed, a.export + ".zip")
        if sha(zpath) != m["outputs"]["zip_sha256"]:
            probs.append("zip sha256")
        with zipfile.ZipFile(zpath) as z:
            names = {n.split("/", 1)[1] for n in z.namelist() if n.endswith(".txt")}
            if names != {f["path"] for f in m["files"]}:
                probs.append("zip file list")
            for f in m["files"]:
                if hashlib.sha256(z.read(a.export + "/" + f["path"])).hexdigest() != f["sha256"]:
                    probs.append("zip:" + f["path"])
        ok &= not probs
        print("EXPORT %s: %d label files (%d empty), %d boxes, zip entries ok, problems %d %s"
              % (a.export, len(m["files"]), sum(1 for f in m["files"] if f["bytes"] == 0),
                 sum(f["boxes"] for f in m["files"]), len(probs), probs[:5]))
    print("RESULT:", "OK" if ok else "PROBLEMS FOUND")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
