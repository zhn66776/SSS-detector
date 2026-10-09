(() => {
  'use strict';
  const el = id => document.getElementById(id);
  const mapCanvas = el('map');
  const mapCtx = mapCanvas.getContext('2d', {alpha: false, desynchronized: true});
  const cropCanvas = el('crop');
  const cropCtx = cropCanvas.getContext('2d', {alpha: false});
  const COLORS = ['#ff5d5d','#33c9bd','#ffd166','#6ea8fe','#cf77e8','#fb923c','#86d35a','#d4a373'];
  const state = {
    config: null, dataset: 'moray', area: 'owf', centerE: 0, centerN: 0, mpp: 1,
    contacts: [], matched: 0, selected: null, enabled: new Set(), categoryCounts: {},
    images: new Map(), pending: new Set(), drag: null, cropCenter: null, contactTimer: null,
    generation: 0
  };

  function dataset() { return state.config.datasets[state.dataset]; }
  function canvasSize(canvas) {
    const dpr = Math.min(devicePixelRatio || 1, 2);
    const rect = canvas.getBoundingClientRect();
    const w = Math.max(1, Math.round(rect.width * dpr));
    const h = Math.max(1, Math.round(rect.height * dpr));
    if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
    return {w, h, dpr};
  }
  function worldToScreen(e, n) {
    const {w, h} = canvasSize(mapCanvas);
    return [(e - state.centerE) / state.mpp + w / 2, (state.centerN - n) / state.mpp + h / 2];
  }
  function screenToWorld(x, y) {
    const {w, h} = canvasSize(mapCanvas);
    return [state.centerE + (x - w / 2) * state.mpp, state.centerN - (y - h / 2) * state.mpp];
  }
  function viewBounds() {
    const {w, h} = canvasSize(mapCanvas);
    return [state.centerE - w * state.mpp / 2, state.centerN - h * state.mpp / 2,
      state.centerE + w * state.mpp / 2, state.centerN + h * state.mpp / 2];
  }
  function fit() {
    const ds = dataset(), b = ds.bounds, {w, h} = canvasSize(mapCanvas);
    state.centerE = (b.left + b.right) / 2;
    state.centerN = (b.bottom + b.top) / 2;
    state.mpp = Math.max((b.right - b.left) / (w * .92), (b.top - b.bottom) / (h * .92));
    state.mpp = Math.max(ds.native_resolution_m / 8, state.mpp);
    viewChanged();
  }
  function clampView() {
    const ds = dataset();
    state.mpp = Math.max(ds.native_resolution_m / 8, Math.min(state.mpp, Math.max(ds.bounds.right-ds.bounds.left, ds.bounds.top-ds.bounds.bottom)));
  }
  function viewChanged() {
    clampView();
    draw();
    clearTimeout(state.contactTimer);
    state.contactTimer = setTimeout(loadContacts, 120);
  }
  function tileLevel() {
    const ds = dataset();
    return Math.max(0, Math.min(ds.max_zoom, Math.round(ds.max_zoom - Math.log2(state.mpp / ds.native_resolution_m))));
  }
  function tileKey(z,x,y) { return `${state.dataset}:${z}:${x}:${y}`; }
  function requestTile(z,x,y) {
    const key = tileKey(z,x,y);
    if (state.images.has(key) || state.pending.has(key)) return;
    state.pending.add(key);
    const image = new Image();
    image.onload = () => {
      state.pending.delete(key); state.images.set(key, image);
      while (state.images.size > 180) state.images.delete(state.images.keys().next().value);
      draw();
    };
    image.onerror = () => { state.pending.delete(key); };
    image.src = `/api/tile?dataset=${state.dataset}&z=${z}&x=${x}&y=${y}`;
  }
  function drawTiles() {
    const ds = dataset();
    if (ds.status !== 'ready') return;
    const z = tileLevel();
    const resolution = ds.native_resolution_m * 2 ** (ds.max_zoom-z);
    const span = 256 * resolution;
    const b = viewBounds();
    const x0 = Math.max(0, Math.floor((b[0]-ds.bounds.left)/span));
    const x1 = Math.floor((Math.min(b[2],ds.bounds.right)-ds.bounds.left)/span);
    const y0 = Math.max(0, Math.floor((ds.bounds.top-b[3])/span));
    const y1 = Math.floor((ds.bounds.top-Math.max(b[1],ds.bounds.bottom))/span);
    for (let y=y0; y<=y1; y++) for (let x=x0; x<=x1; x++) {
      const key=tileKey(z,x,y), image=state.images.get(key);
      const [sx,sy]=worldToScreen(ds.bounds.left+x*span, ds.bounds.top-y*span);
      const size=span/state.mpp;
      if (image) mapCtx.drawImage(image, Math.round(sx), Math.round(sy), Math.ceil(size)+1, Math.ceil(size)+1);
      else requestTile(z,x,y);
    }
  }
  function categoryColor(name) {
    const names=Object.keys(state.categoryCounts).sort();
    return COLORS[Math.max(0,names.indexOf(name)) % COLORS.length];
  }
  function drawContacts() {
    for (const feature of state.contacts) {
      const [x,y]=worldToScreen(feature.e,feature.n);
      if (feature.kind==='cluster') {
        const radius=Math.min(20,7+Math.log2(feature.count)*2);
        mapCtx.beginPath(); mapCtx.arc(x,y,radius,0,Math.PI*2);
        mapCtx.fillStyle='rgba(13,28,39,.88)'; mapCtx.fill();
        mapCtx.strokeStyle='#fff'; mapCtx.lineWidth=1; mapCtx.stroke();
        mapCtx.fillStyle='#fff'; mapCtx.font='600 11px -apple-system,sans-serif'; mapCtx.textAlign='center'; mapCtx.textBaseline='middle'; mapCtx.fillText(feature.count,x,y);
      } else {
        const selected=state.selected && state.selected.uid===feature.uid;
        mapCtx.beginPath(); mapCtx.arc(x,y,selected?7:4.5,0,Math.PI*2);
        mapCtx.fillStyle=categoryColor(feature.description); mapCtx.fill();
        mapCtx.lineWidth=selected?3:1; mapCtx.strokeStyle=selected?'#fff':'rgba(0,0,0,.75)'; mapCtx.stroke();
      }
    }
  }
  function draw() {
    const {w,h}=canvasSize(mapCanvas);
    mapCtx.fillStyle='#efefeb'; mapCtx.fillRect(0,0,w,h);
    drawTiles(); drawContacts(); updateScale();
  }
  function updateScale() {
    const choices=[1,2,5];
    const target=110*state.mpp;
    const power=10**Math.floor(Math.log10(target));
    let metres=power;
    for (const factor of choices) if (factor*power<=target) metres=factor*power;
    el('scale').querySelector('span').style.width=`${metres/state.mpp}px`;
    el('scale').querySelector('b').textContent=metres>=1000?`${(metres/1000).toLocaleString()} km`:`${metres.toLocaleString()} m`;
  }
  async function loadContacts() {
    const b=viewBounds();
    const includeMagnetic=el('include-magnetic').checked;
    const params=new URLSearchParams({dataset:state.dataset,area:state.area,bbox:b.join(','),cluster_m:String(Math.max(0,state.mpp*34)),include_magnetic:includeMagnetic?'1':'0',categories_active:'1'});
    const q=el('search').value.trim(); if(q) params.set('q',q);
    for(const category of state.enabled) params.append('category',category);
    if(includeMagnetic) params.append('category','Magnetic Anomaly');
    const generation=++state.generation;
    try {
      const response=await fetch(`/api/contacts?${params}`); const data=await response.json();
      if(generation!==state.generation) return;
      state.contacts=data.features; state.matched=data.matched;
      if(q && data.matched===1 && data.features.length===1 && data.features[0].kind==='contact') {
        state.centerE=data.features[0].e; state.centerN=data.features[0].n;
      }
      el('contact-count').textContent=`${data.matched.toLocaleString()} contacts in view · ${data.returned.toLocaleString()} markers/clusters`;
      draw();
    } catch(error) { el('status').textContent=`Contact load failed: ${error}`; }
  }
  function populateCategories() {
    const summary=dataset().summary[state.area];
    state.categoryCounts=summary.descriptions;
    state.enabled=new Set(Object.keys(state.categoryCounts).filter(name => name!=='Magnetic Anomaly'));
    const host=el('categories'); host.replaceChildren();
    Object.entries(state.categoryCounts).sort((a,b)=>b[1]-a[1]).forEach(([name,count])=>{
      if(name==='Magnetic Anomaly') return;
      const label=document.createElement('label'); label.className='category-row';
      const input=document.createElement('input'); input.type='checkbox'; input.checked=true;
      input.addEventListener('change',()=>{ input.checked?state.enabled.add(name):state.enabled.delete(name); loadContacts(); draw(); });
      const swatch=document.createElement('i'); swatch.className='swatch'; swatch.style.background=categoryColor(name);
      const text=document.createElement('span'); text.textContent=name;
      const value=document.createElement('span'); value.className='category-count'; value.textContent=count.toLocaleString();
      label.append(input,swatch,text,value); host.append(label);
    });
    el('magnetic-row').hidden=!Object.hasOwn(state.categoryCounts,'Magnetic Anomaly');
  }
  function populateAreas() {
    const area=el('area'); area.replaceChildren();
    for(const item of dataset().areas) area.add(new Option(item.label,item.id));
    state.area=dataset().default_area; area.value=state.area; populateCategories();
  }
  function updateDatasetStatus() {
    const ds=dataset(), message=el('map-message'), status=el('status');
    status.textContent=ds.status_message; status.classList.toggle('warn',ds.status!=='ready');
    message.hidden=ds.status==='ready'; message.textContent=ds.status_message;
  }
  function selectDataset(id) {
    state.dataset=id; state.images.clear(); state.pending.clear(); state.selected=null; state.cropCenter=null;
    el('detail-content').hidden=true; el('detail-empty').hidden=false;
    el('coordinates').textContent='E — · N —';
    populateAreas(); updateDatasetStatus(); fit();
  }
  async function selectContact(uid) {
    const response=await fetch(`/api/contact/${encodeURIComponent(uid)}`); const contact=await response.json();
    if(!response.ok) return;
    state.selected=contact; state.cropCenter={e:contact.e,n:contact.n};
    el('detail-empty').hidden=true; el('detail-content').hidden=false;
    renderContactInfo(contact); loadCrop(); draw();
  }
  function formatValue(value,unit='') { return value===null||value===undefined||value===''?'—':`${Number.isFinite(value)?value.toLocaleString():value}${unit}`; }
  function escapeHtml(value) { const div=document.createElement('div'); div.textContent=String(value??''); return div.innerHTML; }
  function renderContactInfo(c) {
    const fields=[['Easting',formatValue(c.e,' m')],['Northing',formatValue(c.n,' m')],['Length',formatValue(c.length_m,' m')],['Width',formatValue(c.width_m,' m')],['Height',formatValue(c.height_m,' m')],['Magnitude',formatValue(c.magnitude_nt_m,' nT/m')],['Geometry',c.geometry],['Raster',c.raster_id||'outside raster']];
    const raw=Object.entries(c.raw||{}).map(([k,v])=>`<tr><th>${escapeHtml(k)}</th><td>${escapeHtml(v)}</td></tr>`).join('');
    el('contact-info').innerHTML=`<div class="contact-title">${escapeHtml(c.id)}</div><div class="contact-description">${escapeHtml(c.description)}</div><div class="info-grid">${fields.map(([k,v])=>`<div><span>${k}</span><b>${escapeHtml(v)}</b></div>`).join('')}</div><div class="raw-title">Complete source row</div><table class="raw-table">${raw}<tr><th>Source file</th><td>${escapeHtml(c.source_file)}</td></tr><tr><th>Source row</th><td>${c.source_row}</td></tr></table>`;
  }
  async function loadCrop() {
    if(!state.selected||!state.cropCenter) return;
    const loading=el('crop-loading'); loading.hidden=false;
    const span=Number(el('crop-span').value), {e,n}=state.cropCenter;
    const url=`/api/crop?dataset=${state.dataset}&e=${e}&n=${n}&span=${span}`;
    const started=performance.now();
    try {
      const response=await fetch(url); if(!response.ok) throw new Error((await response.json()).error||response.statusText);
      const blob=await response.blob(); const bitmap=await createImageBitmap(blob);
      const {w,h}=canvasSize(cropCanvas); cropCtx.imageSmoothingEnabled=false; cropCtx.clearRect(0,0,w,h); cropCtx.drawImage(bitmap,0,0,w,h);
      const latency=performance.now()-started;
      const native=response.headers.get('X-Native-Resolution-M'); const pixels=response.headers.get('X-Crop-Pixels');
      el('crop-caption').textContent=`${span} m source window · ${native} m native pixels · ${pixels}×${pixels} source samples · ${latency.toFixed(0)} ms`;
      const dx=(state.selected.e-e)/span*100, dy=(n-state.selected.n)/span*100, crosshair=document.querySelector('.crosshair');
      crosshair.style.left=`${50+dx}%`; crosshair.style.top=`${50+dy}%`;
      loading.hidden=true;
    } catch(error) { loading.textContent=`Crop unavailable: ${error.message}`; loading.hidden=false; }
  }
  function markerAt(x,y) {
    let best=null,bestDistance=18;
    for(const feature of state.contacts) { const [sx,sy]=worldToScreen(feature.e,feature.n); const d=Math.hypot(sx-x,sy-y); if(d<bestDistance){best=feature;bestDistance=d;} }
    return best;
  }
  function zoom(factor, x=null, y=null) {
    const {w,h}=canvasSize(mapCanvas); x??=w/2; y??=h/2;
    const before=screenToWorld(x,y); state.mpp*=factor; clampView(); const after=screenToWorld(x,y);
    state.centerE+=before[0]-after[0]; state.centerN+=before[1]-after[1]; viewChanged();
  }
  mapCanvas.addEventListener('wheel',event=>{ event.preventDefault(); const r=mapCanvas.getBoundingClientRect(),d=Math.min(devicePixelRatio||1,2); zoom(event.deltaY>0?1.35:1/1.35,(event.clientX-r.left)*d,(event.clientY-r.top)*d); },{passive:false});
  mapCanvas.addEventListener('pointerdown',event=>{ mapCanvas.setPointerCapture(event.pointerId); state.drag={x:event.clientX,y:event.clientY,e:state.centerE,n:state.centerN,moved:false}; });
  mapCanvas.addEventListener('pointermove',event=>{
    const r=mapCanvas.getBoundingClientRect(),d=Math.min(devicePixelRatio||1,2), [e,n]=screenToWorld((event.clientX-r.left)*d,(event.clientY-r.top)*d);
    el('coordinates').textContent=`E ${e.toFixed(1)} · N ${n.toFixed(1)} · ${state.mpp.toFixed(2)} m/px`;
    if(!state.drag) return; const dx=(event.clientX-state.drag.x)*d,dy=(event.clientY-state.drag.y)*d; if(Math.hypot(dx,dy)>3)state.drag.moved=true;
    state.centerE=state.drag.e-dx*state.mpp; state.centerN=state.drag.n+dy*state.mpp; draw();
  });
  mapCanvas.addEventListener('pointerup',event=>{
    const drag=state.drag; state.drag=null;
    if(drag?.moved){viewChanged();return;}
    const r=mapCanvas.getBoundingClientRect(),d=Math.min(devicePixelRatio||1,2),feature=markerAt((event.clientX-r.left)*d,(event.clientY-r.top)*d);
    if(!feature)return; if(feature.kind==='cluster'){state.centerE=feature.e;state.centerN=feature.n;zoom(.4);}else selectContact(feature.uid);
  });
  const cropWrap=el('crop-wrap'); let cropDrag=null;
  cropWrap.addEventListener('pointerdown',event=>{cropWrap.setPointerCapture(event.pointerId);cropDrag={x:event.clientX,y:event.clientY,e:state.cropCenter.e,n:state.cropCenter.n};cropWrap.classList.add('dragging');});
  cropWrap.addEventListener('pointermove',event=>{if(!cropDrag)return;const span=Number(el('crop-span').value),r=cropWrap.getBoundingClientRect();state.cropCenter={e:cropDrag.e-(event.clientX-cropDrag.x)/r.width*span,n:cropDrag.n+(event.clientY-cropDrag.y)/r.height*span};});
  cropWrap.addEventListener('pointerup',()=>{cropDrag=null;cropWrap.classList.remove('dragging');loadCrop();});
  cropWrap.addEventListener('wheel',event=>{event.preventDefault();const select=el('crop-span'),values=[25,50,100,200,400],current=values.indexOf(Number(select.value)),next=Math.max(0,Math.min(values.length-1,current+(event.deltaY>0?1:-1)));select.value=values[next];loadCrop();},{passive:false});
  el('crop-span').addEventListener('change',loadCrop);
  el('crop-reset').addEventListener('click',()=>{state.cropCenter={e:state.selected.e,n:state.selected.n};loadCrop();});
  el('zoom-in').addEventListener('click',()=>zoom(.5)); el('zoom-out').addEventListener('click',()=>zoom(2)); el('fit').addEventListener('click',fit);
  el('filters-toggle').addEventListener('click',()=>{const p=el('filters');p.classList.toggle('collapsed');el('filters-toggle').textContent=p.classList.contains('collapsed')?'Show':'Hide';});
  el('dataset').addEventListener('change',event=>selectDataset(event.target.value));
  el('area').addEventListener('change',event=>{state.area=event.target.value;state.selected=null;state.cropCenter=null;el('detail-content').hidden=true;el('detail-empty').hidden=false;populateCategories();loadContacts();draw();});
  el('include-magnetic').addEventListener('change',loadContacts);
  el('search').addEventListener('input',()=>{clearTimeout(state.contactTimer);state.contactTimer=setTimeout(loadContacts,180);});
  new ResizeObserver(()=>{if(!state.config)return;draw();loadContacts();}).observe(el('map-wrap'));

  async function init() {
    const response=await fetch('/api/config'); state.config=await response.json();
    const select=el('dataset'); for(const [id,item] of Object.entries(state.config.datasets)) select.add(new Option(item.label,id));
    select.value=state.dataset; selectDataset(state.dataset);
  }
  init().catch(error=>{el('status').textContent=`Startup failed: ${error}`;el('status').classList.add('warn');});
})();
