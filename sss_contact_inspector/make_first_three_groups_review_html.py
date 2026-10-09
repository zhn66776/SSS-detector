#!/usr/bin/env python3
"""Build an offline review map for the first three validation groups."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
BATCH = ROOT / "yolo_manual_annotation_grid_150px_stride100_v1_cleaned" / "validation_annotation_batches_1000"
CONTACTS = ROOT / "data" / "contacts.json"
HTML_OUT = BATCH / "review_first_three_groups.html"
SUMMARY_OUT = BATCH / "review_first_three_groups_summary.json"

LEFT_E = 477323.46875
TOP_N = 6568269.5
PX_M = 0.1
PATCH = 150


def finite(v: Any) -> float | None:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def world_to_pixel(e: float, n: float) -> tuple[float, float]:
    return (e - LEFT_E) / PX_M, (TOP_N - n) / PX_M


def inside_patch(px: float, py: float, x0: int, y0: int) -> bool:
    return x0 <= px < x0 + PATCH and y0 <= py < y0 + PATCH


def load_patches() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    patches: list[dict[str, Any]] = []
    stats: dict[str, Any] = {}
    for number in (1, 2, 3):
        group = f"group_{number:02d}"
        group_dir = BATCH / group
        with (group_dir / "patch_index.csv").open(encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        positive_images = 0
        boxes = 0
        for row in rows:
            image_name = Path(row["file_relative_path"]).name
            annotation_path = group_dir / "images" / (Path(image_name).stem + ".json")
            annotations: list[dict[str, Any]] = []
            if annotation_path.exists():
                document = json.loads(annotation_path.read_text(encoding="utf-8"))
                for shape in document.get("shapes", []):
                    points = shape.get("points", [])
                    if shape.get("shape_type") != "rectangle" or len(points) != 2:
                        continue
                    x1, y1 = map(float, points[0])
                    x2, y2 = map(float, points[1])
                    annotations.append({
                        "label": str(shape.get("label", "")),
                        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                        "world_e": LEFT_E + (int(row["x0"]) + (x1 + x2) / 2) * PX_M,
                        "world_n": TOP_N - (int(row["y0"]) + (y1 + y2) / 2) * PX_M,
                    })
            positive_images += bool(annotations)
            boxes += len(annotations)
            patches.append({
                "group": group,
                "patch_id": row["patch_id"],
                "x0": int(row["x0"]),
                "y0": int(row["y0"]),
                "x1": int(row["x1"]),
                "y1": int(row["y1"]),
                "centre_e": float(row["centre_easting"]),
                "centre_n": float(row["centre_northing"]),
                "annotation_order": int(row["annotation_order"]),
                "within_group_order": int(row["within_group_order"]),
                "image": f"{group}/images/{image_name}",
                "positive": bool(annotations),
                "bboxes": annotations,
                "official": [],
            })
        stats[group] = {
            "images": len(rows),
            "positive_images": positive_images,
            "negative_images": len(rows) - positive_images,
            "bbox_total": boxes,
        }
    patches.sort(key=lambda p: (p["annotation_order"], p["within_group_order"]))
    return patches, stats


def official_overlay(patches: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    source = json.loads(CONTACTS.read_text(encoding="utf-8"))
    by_grid = {(p["x0"], p["y0"]): p for p in patches}
    overlay: list[dict[str, Any]] = []
    point_count = 0
    line_count = 0

    def matching_patches(px: float, py: float) -> list[dict[str, Any]]:
        kx = math.floor(px / 100) * 100
        ky = math.floor(py / 100) * 100
        found = []
        for x0 in (kx - 100, kx):
            for y0 in (ky - 100, ky):
                p = by_grid.get((x0, y0))
                if p is not None and inside_patch(px, py, x0, y0):
                    found.append(p)
        return found

    for row in source:
        if row.get("dataset") != "costa":
            continue
        if row.get("geometry") == "point":
            e, n = finite(row.get("e")), finite(row.get("n"))
            if e is None or n is None:
                continue
            px, py = world_to_pixel(e, n)
            covered = matching_patches(px, py)
            if not covered:
                continue
            item = {
                "kind": "point",
                "uid": str(row.get("uid", "")),
                "id": str(row.get("id", "")),
                "description": str(row.get("description", "")),
                "e": e, "n": n,
            }
            overlay.append(item)
            point_count += 1
            for p in covered:
                p["official"].append(item)
        elif row.get("geometry") == "line":
            values = [finite(row.get(k)) for k in ("start_e", "start_n", "end_e", "end_n")]
            if any(v is None for v in values):
                continue
            se, sn, ee, en = values
            mid_e, mid_n = (se + ee) / 2, (sn + en) / 2
            covered = []
            for e, n in ((se, sn), (ee, en), (mid_e, mid_n)):
                for p in matching_patches(*world_to_pixel(e, n)):
                    if p not in covered:
                        covered.append(p)
            if not covered:
                continue
            item = {
                "kind": "line",
                "uid": str(row.get("uid", "")),
                "id": str(row.get("id", "")),
                "description": str(row.get("description", "")),
                "start_e": se, "start_n": sn, "end_e": ee, "end_n": en,
            }
            overlay.append(item)
            line_count += 1
            for p in covered:
                p["official"].append(item)
    return overlay, {
        "point_records_in_patch_footprints": point_count,
        "line_records_by_endpoint_or_midpoint": line_count,
    }


def make_html(patches: list[dict[str, Any]], stats: dict[str, Any], official: list[dict[str, Any]], official_stats: dict[str, int]) -> str:
    min_e = min(p["centre_e"] for p in patches) - 20
    max_e = max(p["centre_e"] for p in patches) + 20
    min_n = min(p["centre_n"] for p in patches) - 20
    max_n = max(p["centre_n"] for p in patches) + 20
    width = 1100
    height = max(650, int(width * (max_n - min_n) / (max_e - min_e)))

    def sx(e: float) -> float:
        return (e - min_e) / (max_e - min_e) * width

    def sy(n: float) -> float:
        return (max_n - n) / (max_n - min_n) * height

    for p in patches:
        p["svg_x"] = sx(LEFT_E + p["x0"] * PX_M)
        p["svg_y"] = sy(TOP_N - p["y0"] * PX_M)
        p["svg_w"] = PATCH * PX_M / (max_e - min_e) * width
        p["svg_h"] = PATCH * PX_M / (max_n - min_n) * height
        for b in p["bboxes"]:
            b["svg_x"] = sx(b["world_e"])
            b["svg_y"] = sy(b["world_n"])
    svg_official = []
    for o in official:
        if o["kind"] == "point":
            svg_official.append({**o, "svg_x": sx(o["e"]), "svg_y": sy(o["n"])})
        else:
            svg_official.append({
                **o, "svg_x1": sx(o["start_e"]), "svg_y1": sy(o["start_n"]),
                "svg_x2": sx(o["end_e"]), "svg_y2": sy(o["end_n"]),
            })
    data = {
        "patches": patches,
        "stats": stats,
        "official": svg_official,
        "official_stats": official_stats,
        "map": {"width": width, "height": height},
    }
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    stats_line = "；".join(
        f"{g}: {v['positive_images']}正/{v['negative_images']}负/{v['bbox_total']}框"
        for g, v in stats.items()
    )
    official_line = (
        f"官方点记录 {official_stats['point_records_in_patch_footprints']} 条；"
        f"线记录 {official_stats['line_records_by_endpoint_or_midpoint']} 条。"
    )
    return """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>前三个 validation group 对照审查</title>
<style>
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;margin:0;background:#111;color:#eee}
header{padding:16px 22px;background:#20252b;position:sticky;top:0;z-index:5}
h1{font-size:20px;margin:0 0 6px}.sub{font-size:13px;color:#b9c0c8}
.layout{display:grid;grid-template-columns:minmax(600px,1fr) 380px;gap:14px;padding:14px}
.panel{background:#1b1f24;border:1px solid #3a4149;border-radius:8px;padding:10px;min-width:0}
#map{width:100%;height:auto;background:#fafafa;cursor:pointer}
.controls{display:flex;gap:12px;flex-wrap:wrap;margin-bottom:8px;font-size:13px}
.legend span{margin-right:14px}.swatch{display:inline-block;width:12px;height:12px;margin-right:4px;vertical-align:-1px;border:1px solid #999}
#preview{width:100%;image-rendering:pixelated;background:#fff;border:1px solid #555}
#info{font-size:13px;line-height:1.5;color:#d8dde2}
button{background:#303943;color:#fff;border:1px solid #59636e;border-radius:4px;padding:5px 8px}
</style></head><body>
<header><h1>前三个 validation group：人工标注与官方 contact list 对照</h1>
<div class="sub">""" + stats_line + "<br>" + official_line + """<br>
官方图层只用于对照，不会生成或修改任何标签。点击地图中的 patch 查看原图。</div></header>
<div class="layout"><section class="panel">
<div class="controls"><label><input id="neg" type="checkbox" checked>显示负样本 patch</label>
<label><input id="official" type="checkbox" checked>显示官方 contacts</label>
<button id="reset">恢复全图</button></div>
<div class="legend"><span><i class="swatch" style="background:#9b59b6"></i>group01</span>
<span><i class="swatch" style="background:#e67e22"></i>group02</span>
<span><i class="swatch" style="background:#16a085"></i>group03</span>
<span><i class="swatch" style="background:#e74c3c"></i>人工正样本</span>
<span><i class="swatch" style="background:#2878c8"></i>官方点/线</span></div>
<svg id="map" viewBox="0 0 """ + str(width) + " " + str(height) + """"></svg>
</section><section class="panel"><img id="preview" alt="点击地图选择 patch">
<div id="info">点击左侧地图中的 patch。</div></section></div>
<script>
const DATA = """ + payload + """;
const svg=document.getElementById("map"), preview=document.getElementById("preview"), info=document.getElementById("info");
const colors={group_01:"#9b59b6",group_02:"#e67e22",group_03:"#16a085"};
function render(){
  let out="";
  DATA.patches.forEach(function(p,i){
    if(!document.getElementById("neg").checked && !p.positive) return;
    const color=colors[p.group]||"#777";
    const stroke=p.positive?"#e74c3c":color;
    const sw=p.positive?2.4:0.45;
    const op=p.positive?0.9:0.2;
    out+='<rect data-i="'+i+'" x="'+p.svg_x+'" y="'+p.svg_y+'" width="'+p.svg_w+'" height="'+p.svg_h+'" fill="'+color+'" fill-opacity="'+op+'" stroke="'+stroke+'" stroke-width="'+sw+'"><title>'+p.group+" "+p.patch_id+"; manual="+p.bboxes.length+"; official="+p.official.length+'</title></rect>';
  });
  if(document.getElementById("official").checked){
    DATA.official.forEach(function(o){
      if(o.kind==="point"){
        out+='<rect x="'+(o.svg_x-2.6)+'" y="'+(o.svg_y-2.6)+'" width="5.2" height="5.2" transform="rotate(45 '+o.svg_x+" "+o.svg_y+')" fill="#2878c8" stroke="#fff" stroke-width=".7"><title>'+o.uid+" "+o.description+'</title></rect>';
      }else{
        out+='<line x1="'+o.svg_x1+'" y1="'+o.svg_y1+'" x2="'+o.svg_x2+'" y2="'+o.svg_y2+'" stroke="#2878c8" stroke-width="2" opacity=".85"><title>'+o.uid+' '+o.description+'</title></line>';
      }
    });
  }
  svg.innerHTML=out;
  svg.querySelectorAll("rect[data-i]").forEach(function(el){el.addEventListener("click",function(){selectPatch(Number(el.dataset.i));});});
}
function selectPatch(i){
  const p=DATA.patches[i];
  preview.src=p.image;
  let s="<b>"+p.group+" / "+p.patch_id+"</b><br>中心 E="+p.centre_e.toFixed(3)+" N="+p.centre_n.toFixed(3);
  s+="<br>人工 bbox："+p.bboxes.length+" 个<br>官方对照记录："+p.official.length+" 条";
  if(p.bboxes.length){s+="<br><br><b>人工框</b><br>";p.bboxes.forEach(function(b,j){s+="#"+(j+1)+" "+b.label+" ["+b.x1.toFixed(1)+","+b.y1.toFixed(1)+"]–["+b.x2.toFixed(1)+","+b.y2.toFixed(1)+"]<br>";});}
  if(p.official.length){s+="<br><b>官方对照</b><br>";p.official.forEach(function(o){s+=o.kind+" "+o.uid+" "+o.description+"<br>";});}
  info.innerHTML=s;
}
document.getElementById("neg").onchange=render;
document.getElementById("official").onchange=render;
document.getElementById("reset").onclick=function(){render();selectPatch(0);};
render();
selectPatch(DATA.patches.findIndex(function(p){return p.positive;}));
</script></body></html>
"""


def main() -> None:
    patches, stats = load_patches()
    official, official_stats = official_overlay(patches)
    HTML_OUT.write_text(make_html(patches, stats, official, official_stats), encoding="utf-8")
    summary = {
        "html": str(HTML_OUT),
        "groups": stats,
        "total_patches": len(patches),
        "total_positive_images": sum(s["positive_images"] for s in stats.values()),
        "total_negative_images": sum(s["negative_images"] for s in stats.values()),
        "total_manual_bboxes": sum(s["bbox_total"] for s in stats.values()),
        "official_overlay": official_stats,
        "official_contact_list_used_for": "visual overlay only; no label generation",
    }
    SUMMARY_OUT.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
