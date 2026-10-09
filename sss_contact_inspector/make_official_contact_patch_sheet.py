#!/usr/bin/env python3
from __future__ import annotations
import csv, json, math
from pathlib import Path
from collections import defaultdict
from PIL import Image, ImageDraw, ImageFont

ROOT=Path(__file__).resolve().parent
BATCH=ROOT/"yolo_manual_annotation_grid_150px_stride100_v1_cleaned"/"validation_annotation_batches_1000"
OUT=BATCH/"official_contact_patches_first_three_groups.png"
CSV_OUT=BATCH/"official_contact_patch_membership_first_three_groups.csv"
LEFT_E,TOP_N,PX_M,PATCH=477323.46875,6568269.5,0.1,150


def font(size):
    for p in ("/System/Library/Fonts/Helvetica.ttc","/Library/Fonts/Arial.ttf"):
        if Path(p).exists():
            try:return ImageFont.truetype(p,size)
            except OSError:pass
    return ImageFont.load_default()


def main():
    patches=[]
    for g in (1,2,3):
        gd=BATCH/f"group_{g:02d}"
        with (gd/"patch_index.csv").open(newline="",encoding="utf-8") as f:
            for r in csv.DictReader(f):
                r["group"]=f"group_{g:02d}"
                patches.append(r)
    lookup={(int(r["x0"]),int(r["y0"])):r for r in patches}
    membership=defaultdict(list)
    contacts=json.loads((ROOT/"data"/"contacts.json").read_text())
    for c in contacts:
        if c.get("dataset")!="costa" or c.get("geometry")!="point":continue
        try:e,n=float(c["e"]),float(c["n"])
        except (TypeError,ValueError):continue
        px=(e-LEFT_E)/PX_M; py=(TOP_N-n)/PX_M
        kx,ky=math.floor(px/100)*100,math.floor(py/100)*100
        for x0 in (kx-100,kx):
            for y0 in (ky-100,ky):
                r=lookup.get((x0,y0))
                if r and x0<=px<x0+PATCH and y0<=py<y0+PATCH:
                    key=r["patch_id"]
                    if not any(m["uid"]==str(c.get("uid","")) for m in membership[key]):
                        membership[key].append({"uid":str(c.get("uid","")),"id":str(c.get("id","")),"description":str(c.get("description","")),"local_x":px-x0,"local_y":py-y0,"e":e,"n":n})
    selected=[r for r in patches if r["patch_id"] in membership]
    selected.sort(key=lambda r:(r["group"],int(r["y0"]),int(r["x0"])))
    cols=5; card_w=230; card_h=235; title_h=90
    rows=math.ceil(len(selected)/cols)
    canvas=Image.new("RGB",(cols*card_w+30,title_h+rows*card_h+24),(245,245,245))
    d=ImageDraw.Draw(canvas); title=font(25); body=font(13); small=font(11)
    d.text((18,15),"Official contact-list patches — first three validation groups",fill=(20,20,20),font=title)
    d.text((18,51),f"{len(selected)} unique patches containing {sum(len(v) for v in membership.values())} official point records; blue cross = official point",fill=(65,65,65),font=body)
    colors={"group_01":(155,89,182),"group_02":(230,126,34),"group_03":(22,160,133)}
    fields=["patch_id","group","official_uid","official_id","description","local_x_px","local_y_px","easting","northing"]
    membership_rows=[]
    for idx,r in enumerate(selected,1):
        col=(idx-1)%cols; rr=(idx-1)//cols
        x=col*card_w+12; y=title_h+rr*card_h
        d.rounded_rectangle((x,y,x+card_w-12,y+card_h-12),radius=8,fill=(255,255,255),outline=colors[r["group"]],width=2)
        src=BATCH/r["group"]/r["file_relative_path"]
        with Image.open(src) as im:
            im=im.convert("L").resize((180,180),Image.Resampling.NEAREST).convert("RGB")
        ox,oy=x+25,y+8; canvas.paste(im,(ox,oy))
        for m in membership[r["patch_id"]]:
            mx=ox+(m["local_x"]/PATCH)*180; my=oy+(m["local_y"]/PATCH)*180
            d.line((mx-8,my,mx+8,my),fill=(25,105,220),width=3)
            d.line((mx,my-8,mx,my+8),fill=(25,105,220),width=3)
            d.ellipse((mx-4,my-4,mx+4,my+4),outline=(255,255,255),width=1)
            membership_rows.append({"patch_id":r["patch_id"],"group":r["group"],"official_uid":m["uid"],"official_id":m["id"],"description":m["description"],"local_x_px":f"{m['local_x']:.3f}","local_y_px":f"{m['local_y']:.3f}","easting":f"{m['e']:.3f}","northing":f"{m['n']:.3f}"})
        d.rectangle((ox,oy,ox+179,oy+179),outline=(75,75,75),width=1)
        label=f"{idx:02d} {r['group']} {r['patch_id'].replace('costa_val_','')}"
        d.text((x+10,y+192),label,fill=(20,20,20),font=small)
        ids=", ".join(m["id"] or m["uid"] for m in membership[r["patch_id"]])
        d.text((x+10,y+209),"official: "+ids[:31],fill=(35,95,180),font=small)
    canvas.save(OUT,format="PNG",optimize=True)
    with CSV_OUT.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(membership_rows)
    print(json.dumps({"png":str(OUT),"csv":str(CSV_OUT),"unique_patches":len(selected),"official_point_records":len(membership_rows)},indent=2))


if __name__=="__main__":main()
