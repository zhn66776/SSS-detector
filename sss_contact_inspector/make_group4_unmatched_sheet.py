#!/usr/bin/env python3
from __future__ import annotations
import csv,json
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont

ROOT=Path(__file__).resolve().parent
BATCH=ROOT/"yolo_manual_annotation_grid_150px_stride100_v1_cleaned"/"validation_annotation_batches_1000"
OUT=BATCH/"group4_unmatched_official_S1642_S1755.png"
CSV_OUT=BATCH/"group4_unmatched_official_S1642_S1755.csv"
LEFT_E,TOP_N,PX_M,PATCH=477323.46875,6568269.5,0.1,150
TARGETS={"S1642":(483808.0,6563131.6),"S1755":(484188.4,6563212.9)}


def font(size):
    for p in ("/System/Library/Fonts/Helvetica.ttc","/Library/Fonts/Arial.ttf"):
        if Path(p).exists():
            try:return ImageFont.truetype(p,size)
            except OSError:pass
    return ImageFont.load_default()


def main():
    with (BATCH/"group_04"/"patch_index.csv").open(newline="",encoding="utf-8") as f:
        rows=list(csv.DictReader(f))
    by_id={r["patch_id"]:r for r in rows}
    cards=[]
    membership=[]
    for official_id,(e,n) in TARGETS.items():
        px=(e-LEFT_E)/PX_M;py=(TOP_N-n)/PX_M
        kx,ky=int(px//100)*100,int(py//100)*100
        for x0 in (kx-100,kx):
            for y0 in (ky-100,ky):
                patch_id=next((r["patch_id"] for r in rows if int(r["x0"])==x0 and int(r["y0"])==y0 and x0<=px<x0+PATCH and y0<=py<y0+PATCH),None)
                if patch_id is None:continue
                r=by_id[patch_id]
                cards.append((official_id,e,n,px-x0,py-y0,r))
                membership.append({"official_id":official_id,"easting":e,"northing":n,"patch_id":patch_id,"local_x_px":px-x0,"local_y_px":py-y0})
    # deterministic order: S1642 then S1755, raster x order
    cards.sort(key=lambda x:(x[0],int(x[5]["x0"])))
    with CSV_OUT.open("w",newline="",encoding="utf-8") as f:
        fields=["official_id","easting","northing","patch_id","local_x_px","local_y_px"]
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(membership)
    cols=2;cw=520;ch=520;title_h=100
    canvas=Image.new("RGB",(cols*cw+20,title_h+((len(cards)+1)//2)*ch+20),(245,245,245))
    d=ImageDraw.Draw(canvas)
    d.text((18,16),"Group 04 unmatched official contacts",fill=(20,20,20),font=font(28))
    d.text((18,57),"Blue cross = official contact-list coordinate; red boxes would be saved manual bboxes (none around these points)",fill=(70,70,70),font=font(15))
    for i,(oid,e,n,lx,ly,r) in enumerate(cards):
        x=(i%cols)*cw+12;y=title_h+(i//cols)*ch
        d.rounded_rectangle((x,y,x+cw-14,y+ch-14),radius=10,fill=(255,255,255),outline=(215,70,60),width=2)
        with Image.open(BATCH/"group_04"/r["file_relative_path"]).convert("L") as im:
            im=im.resize((400,400),Image.Resampling.NEAREST).convert("RGB")
        ox,oy=x+60,y+10;canvas.paste(im,(ox,oy))
        mx=ox+lx/PATCH*400;my=oy+ly/PATCH*400
        d.line((mx-12,my,mx+12,my),fill=(25,105,220),width=4);d.line((mx,my-12,mx,my+12),fill=(25,105,220),width=4)
        d.ellipse((mx-5,my-5,mx+5,my+5),outline=(255,255,255),width=2)
        d.rectangle((ox,oy,ox+399,oy+399),outline=(70,70,70),width=1)
        d.text((x+18,y+420),f"{oid}  E={e:.1f}  N={n:.1f}",fill=(20,20,20),font=font(16))
        d.text((x+18,y+445),f"{r['patch_id']}  local=({lx:.1f},{ly:.1f}) px",fill=(30,90,190),font=font(14))
    canvas.save(OUT,format="PNG",optimize=True)
    print(json.dumps({"png":str(OUT),"csv":str(CSV_OUT),"cards":len(cards)},indent=2))


if __name__=="__main__":main()
