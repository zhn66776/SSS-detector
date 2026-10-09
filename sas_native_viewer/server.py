"""Local, on-demand viewer for the original 3 cm SAS pixels."""

from __future__ import annotations

from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
import json
import threading
import warnings

from PIL import Image
import tifffile

from render import ROOT, SOURCE, tile, to_gray


with warnings.catch_warnings():
    warnings.filterwarnings('ignore', message='.*GDAL_NODATA.*')
    TIFF = tifffile.TiffFile(SOURCE)
PAGE = TIFF.pages[0]
W, H = PAGE.imagewidth, PAGE.imagelength
COLS = (PAGE.imagewidth + PAGE.tilewidth - 1) // PAGE.tilewidth
ROWS = (PAGE.imagelength + PAGE.tilelength - 1) // PAGE.tilelength
CACHE: OrderedDict[int, bytes] = OrderedDict()
LOCK = threading.Lock()


def png_for_tile(index: int) -> bytes:
    with LOCK:
        if index in CACHE:
            CACHE.move_to_end(index)
            return CACHE[index]
    with SOURCE.open('rb') as file:
        a = tile(PAGE, file, index)
    im = Image.fromarray(to_gray(a))
    im = im.crop((0, 0, min(2048, PAGE.imagewidth - (index % COLS) * 2048), min(2048, PAGE.imagelength - (index // COLS) * 2048)))
    buf = BytesIO()
    im.save(buf, format='PNG', optimize=True)
    data = buf.getvalue()
    with LOCK:
        CACHE[index] = data
        CACHE.move_to_end(index)
        while len(CACHE) > 4:
            CACHE.popitem(last=False)
    return data


HTML = '''<!doctype html><html lang="zh"><meta charset="utf-8"><title>Galápagos SAS · 原像素查看</title>
<style>body{font:15px system-ui;background:#151515;color:#eee;margin:20px}h1{font-size:20px}p{color:#bbb}
.layout{display:grid;grid-template-columns:minmax(360px,1fr) minmax(550px,1fr);gap:18px}
.panel{background:#222;border:1px solid #444;padding:12px;min-width:0}
#overview{display:block;width:100%;height:auto;image-rendering:auto;cursor:crosshair;background:#000}
#detail{height:75vh;overflow:auto;background:#000}#native{display:block;max-width:none;image-rendering:pixelated}
@media(max-width:1000px){.layout{grid-template-columns:1fr}}
</style><h1>Galápagos InSAS · 3 cm 原像素查看</h1>
<p>左边点击测区；右边逐块显示未经缩小的 2048×2048 原始像素。右边可滚动查看。灰度仅做固定范围显示（−35 至 35）；黑色为无数据。</p>
<div class="layout"><div class="panel"><strong>测区概览（缩小 16 倍）</strong><img id="overview" src="/overview.png"></div>
<div class="panel"><strong id="status">点击左侧选择位置</strong><div id="detail"><img id="native"></div></div></div>
<script>const W=44760,H=34924,COLS=22;const ov=document.querySelector('#overview');
ov.onclick=e=>{let r=ov.getBoundingClientRect(),x=Math.floor((e.clientX-r.left)/r.width*W),y=Math.floor((e.clientY-r.top)/r.height*H);
let col=Math.floor(x/2048),row=Math.floor(y/2048),id=row*COLS+col;
document.querySelector('#status').textContent=`原像素块 ${id} · 列 ${col} 行 ${row} · 像素 (${col*2048}, ${row*2048})`;
let im=document.querySelector('#native');im.src='/tile/'+id;im.onload=()=>{let d=document.querySelector('#detail');d.scrollLeft=x%2048-d.clientWidth/2;d.scrollTop=y%2048-d.clientHeight/2};};
</script></html>'''


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        try:
            if self.path == '/':
                body, mime = HTML.encode(), 'text/html; charset=utf-8'
            elif self.path == '/overview.png':
                body, mime = (ROOT / 'overview_16x.png').read_bytes(), 'image/png'
            elif self.path.startswith('/tile/'):
                index = int(self.path.split('/')[-1])
                if not 0 <= index < COLS * ROWS:
                    raise ValueError('tile index')
                body, mime = png_for_tile(index), 'image/png'
            elif self.path == '/meta':
                body, mime = json.dumps(dict(width=W, height=H, tile=2048, cols=COLS, rows=ROWS)).encode(), 'application/json'
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header('Content-Type', mime)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'private, max-age=3600')
            self.end_headers()
            self.wfile.write(body)
        except (ValueError, IndexError):
            self.send_error(400)


if __name__ == '__main__':
    server = ThreadingHTTPServer(('127.0.0.1', 8767), Handler)
    print('http://127.0.0.1:8767', flush=True)
    server.serve_forever()
