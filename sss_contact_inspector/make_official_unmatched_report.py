#!/usr/bin/env python3
from __future__ import annotations
import csv,json,math
from collections import defaultdict
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont

ROOT=Path(__file__).resolve().parent
BATCH=ROOT/"yolo_manual_annotation_grid_150px_stride100_v1_cleaned"/"validation_annotation_batches_1000"
CONTACTS=ROOT/"data"/"contacts.json"
CSV_OUT=BATCH/"official_unmatched_first_three_groups.csv"
PNG_OUT=BATCH/"official_unmatched_first_three_groups.png"
LEFT_E,TOP_N,PX_M,PATCH=477323.46875,6568269.5,0.1,150


def font(size):
    for p in ("/System/Library/Fonts/Helvetica.ttc","/Library/Fonts/Arial.ttf"):
        if Path(p).exists():
            try:return ImageFont.truetype(p,size)
            except OSError:pass
    return ImageFont.load_default()


def main():
    lookup={}
    for g in (1,2,3):
        gd=BATCH/f"group_{g:02d}"
        with (gd/"patch_index.csv").open(newline="",encoding="utf-8") as f:
            for r in csv.DictReader(f):
                r={**r,"group":f"group_{g:02d}","bboxes":[]}
                ap=gd/"images"/(Path(r["file_relative_path"]).stem+".json")
                if ap.exists():
                    for s in json.loads(ap.read_text()).get("shapes",[]):
                        if s.get("shape_type")=="rectangle" and len(s.get("points",[]))==2:
                            (x1,y1),(x2,y2)=s["points"]
                            r["bboxes"].append((float(min(x1,x2)),float(min(y1,y2)),float(max(x1,x2)),float(max(y1,y2))))
                lookup[(int(r["x0"]),int(r["y0"]))]=r
    unmatched=[]; covered_patches={}
    for c in json.loads(CONTACTS.read_text()):
        if c.get("dataset")!="costa" or c.get("geometry")!="point":continue
        try:e,n=float(c["e"]),float(c["n"])
        except (TypeError,ValueError):continue
        px=(e-LEFT_E)/PX_M; py=(TOP_N-n)/PX_M
        kx,ky=math.floor(px/100)*100,math.floor(py/100)*100
        covered=[]
        for x0 in (kx-100,kx):
            for y0 in (ky-100,ky):
                p=lookup.get((x0,y0))
                if p and x0<=px<x0+PATCH and y0<=py<y0+PATCH and p not in covered:covered.append(p)
        if not covered:continue
        matches=[]; near=[]
        for p in covered:
            lx=px-int(p["x0"]);ly=py-int(p["y0"])
            for b in p["bboxes"]:
                inside=b[0]<=lx<=b[2] and b[1]<=ly<=b[3]
                near.append((math.hypot(lx-(b[0]+b[2])/2,ly-(b[1]+b[3])/2),p["patch_id"]))
                if inside:matches.append(p["patch_id"])
        if matches:continue
        nearest=min(near,key=lambda x:x[0]) if near else (None,None)
        patchids=[p["patch_id"] for p in covered]
        item={"uid":str(c.get("uid","")),"id":str(c.get("id","")),"area":str(c.get("area","")),"description":str(c.get("description","")),"easting":f"{e:.3f}","northing":f"{n:.3f}","covered_patch_count":len(covered),"covered_patch_ids":";".join(patchids),"nearest_bbox_centre_distance_px":"" if nearest[0] is None else f"{nearest[0]:.3f}","nearest_bbox_patch_id":nearest[1] or ""}
        unmatched.append(item)
        for p in covered:
            lx=px-int(p["x0"]);ly=py-int(p["y0"])
            covered_patches.setdefault(p["patch_id"],{"row":p,"points":[]})["points"].append({"id":item["id"],"lx":lx,"ly":ly})
    fields=["uid","id","area","description","easting","northing","covered_patch_count","covered_patch_ids","nearest_bbox_centre_distance_px","nearest_bbox_patch_id"]
    with CSV_OUT.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(unmatched)
    selected=sorted(covered_patches.values(),key=lambda v:(v["row"]["group"],int(v["row"]["y0"]),int(v["row"]["x0"])))
    cols=4;cw=285;ch=270;title_h=85
    canvas=Image.new("RGB",(cols*cw+24,title_h+math.ceil(len(selected)/cols)*ch+20),(245,245,245))
    d=ImageDraw.Draw(canvas);d.text((18,16),"Unmatched official contact-list patches",fill=(20,20,20),font=font(26))
    d.text((18,50),f"{len(unmatched)} official records; {len(selected)} unique covered patches; blue cross = official point, red = manual boxes",fill=(70,70,70),font=font(14))
    for i,item in enumerate(selected,1):
        r=item["row"];x=(i-1)%cols*cw+12;y=title_h+(i-1)//cols*ch
        d.rounded_rectangle((x,y,x+cw-10,y+ch-10),radius=8,fill=(255,255,255),outline=(210,80,70),width=2)
        with Image.open(BATCH/r["group"]/r["file_relative_path"]).convert("L") as im:
            im=im.resize((220,220),Image.Resampling.NEAREST).convert("RGB")
        ox,oy=x+30,y+8;canvas.paste(im,(ox,oy))
        for b in r["bboxes"]:
            d.rectangle((ox+b[0]/PATCH*220,oy+b[1]/PATCH*220,ox+b[2]/PATCH*220,oy+b[3]/PATCH*220),outline=(225,50,45),width=3)
        for p in item["points"]:
            px=ox+p["lx"]/PATCH*220;py=oy+p["ly"]/PATCH*220
            d.line((px-9,py,px+9,py),fill=(20,100,220),width=3);d.line((px,py-9,px,py+9),fill=(20,100,220),width=3)
        d.text((x+10,y+232),f"{i:02d} {r['group']} {r['patch_id'].replace('costa_val_','')}",fill=(20,20,20),font=font(12))
        ids=", ".join(p["id"] for p in item["points"])
        d.text((x+10,y+248),"official: "+ids[:34],fill=(20,90,190),font=font(11))
    canvas.save(PNG_OUT,format="PNG",optimize=True)
    summary={"csv":str(CSV_OUT),"png":str(PNG_OUT),"official_records_in_first3_footprints":28,"matched_records":28-len(unmatched),"unmatched_records":len(unmatched),"unique_unmatched_covered_patches":len(selected),"matching_rule":"official point must fall inside at least one saved manual rectangle in an overlapping patch"}
    print(json.dumps(summary,indent=2))
    print("UNMATCHED_IDS",",".join(x["id"] for x in unmatched))


if __name__=="__main__":main()
