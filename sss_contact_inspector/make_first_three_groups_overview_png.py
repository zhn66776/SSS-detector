#!/usr/bin/env python3
from __future__ import annotations
import csv, json, math
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
BATCH = ROOT / "yolo_manual_annotation_grid_150px_stride100_v1_cleaned" / "validation_annotation_batches_1000"
OUT = BATCH / "review_first_three_groups_overview.png"
LEFT_E, TOP_N, PX_M, PATCH = 477323.46875, 6568269.5, 0.1, 150


def font(size):
    for p in ("/System/Library/Fonts/Helvetica.ttc", "/Library/Fonts/Arial.ttf"):
        if Path(p).exists():
            try:
                return ImageFont.truetype(p, size)
            except OSError:
                pass
    return ImageFont.load_default()


def project(e, n, bounds, area):
    min_e, min_n, max_e, max_n = bounds
    left, top, width, height = area
    scale = min(width / (max_e-min_e), height / (max_n-min_n))
    used_w, used_h = (max_e-min_e)*scale, (max_n-min_n)*scale
    ox, oy = left+(width-used_w)/2, top+(height-used_h)/2
    return ox+(e-min_e)*scale, oy+(max_n-n)*scale


def main():
    patches=[]
    stats={}
    for num in (1,2,3):
        group=f"group_{num:02d}"
        gd=BATCH/group
        with (gd/"patch_index.csv").open(newline="",encoding="utf-8") as f:
            rows=list(csv.DictReader(f))
        positive=0; boxes=0
        for row in rows:
            ap=gd/"images"/(Path(row["file_relative_path"]).stem+".json")
            b=[]
            if ap.exists():
                d=json.loads(ap.read_text())
                for s in d.get("shapes",[]):
                    if s.get("shape_type")=="rectangle" and len(s.get("points",[]))==2:
                        b.append(s)
            positive += bool(b); boxes += len(b)
            patches.append({**row,"group":group,"positive":bool(b),"bboxes":b})
        stats[group]=(len(rows),positive,len(rows)-positive,boxes)
    e_vals=[LEFT_E+int(p["x0"])*PX_M for p in patches]+[LEFT_E+(int(p["x1"])+1)*PX_M for p in patches]
    n_vals=[TOP_N-int(p["y0"])*PX_M for p in patches]+[TOP_N-(int(p["y1"])+1)*PX_M for p in patches]
    bounds=(min(e_vals)-10,min(n_vals)-10,max(e_vals)+10,max(n_vals)+10)
    canvas=Image.new("RGB",(1800,1100),(248,248,248))
    d=ImageDraw.Draw(canvas); title=font(30); sub=font(17); small=font(15)
    d.text((26,18),"前三个 validation group：人工标注与官方 contact list 对照",fill=(20,20,20),font=title)
    d.text((26,58),"紫/橙/绿=group01/02/03；红框=人工正样本 patch；蓝点=官方 contact list 对照（仅可视化）",fill=(70,70,70),font=sub)
    line="；".join(f"{g}: {v[1]}正/{v[2]}负/{v[3]}框" for g,v in stats.items())
    d.text((26,84),line,fill=(70,70,70),font=sub)
    map_area=(24,125,1260,900)
    group_color={"group_01":(155,89,182),"group_02":(230,126,34),"group_03":(22,160,133)}
    for p in patches:
        x0=LEFT_E+int(p["x0"])*PX_M; x1=LEFT_E+(int(p["x1"])+1)*PX_M
        n0=TOP_N-int(p["y0"])*PX_M; n1=TOP_N-(int(p["y1"])+1)*PX_M
        a=project(x0,n0,bounds,map_area); b=project(x1,n1,bounds,map_area)
        col=group_color[p["group"]]; outline=(231,76,60) if p["positive"] else col
        d.rectangle((a[0],a[1],b[0],b[1]),fill=col,outline=outline,width=2 if p["positive"] else 1)
    # Official point records inside the three patch footprints.
    contacts=json.loads((ROOT/"data"/"contacts.json").read_text())
    patch_keys={(int(p["x0"]),int(p["y0"])) for p in patches}
    for c in contacts:
        if c.get("dataset")!="costa" or c.get("geometry")!="point": continue
        try: e,n=float(c["e"]),float(c["n"])
        except (TypeError,ValueError): continue
        px,py=(e-LEFT_E)/PX_M,(TOP_N-n)/PX_M
        kx,ky=math.floor(px/100)*100,math.floor(py/100)*100
        covered=False
        for xx in (kx-100,kx):
            for yy in (ky-100,ky):
                if (xx,yy) in patch_keys and xx<=px<xx+PATCH and yy<=py<yy+PATCH: covered=True
        if covered:
            x,y=project(e,n,bounds,map_area)
            d.polygon([(x,y-5),(x+5,y),(x,y+5),(x-5,y)],fill=(40,120,200),outline=(255,255,255))
    # Add one selected annotated patch at right.
    selected=next((p for p in patches if p["positive"]), patches[0])
    img=Image.open(BATCH/selected["group"]/"images"/Path(selected["file_relative_path"]).name).convert("L").resize((360,360),Image.Resampling.NEAREST).convert("RGB")
    panel=(1320,125,450,900)
    d.rounded_rectangle((panel[0],panel[1],panel[0]+panel[2],panel[1]+panel[3]),radius=12,fill=(30,34,40),outline=(100,105,110),width=2)
    d.text((panel[0]+18,panel[1]+18),"示例正样本",fill=(240,240,240),font=title)
    canvas.paste(img,(panel[0]+45,panel[1]+75))
    d.text((panel[0]+18,panel[1]+455),selected["patch_id"],fill=(240,240,240),font=sub)
    d.text((panel[0]+18,panel[1]+485),f"人工 bbox: {len(selected['bboxes'])} 个",fill=(240,240,240),font=sub)
    d.text((panel[0]+18,panel[1]+525),"蓝色菱形：官方 contact list",fill=(150,200,255),font=sub)
    d.text((panel[0]+18,panel[1]+560),"HTML 中可点击每个 patch 查看原图",fill=(190,190,190),font=sub)
    canvas.save(OUT,format="PNG",optimize=True)
    print(OUT)

if __name__=="__main__":
    main()
