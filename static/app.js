/* ─────────────────────────────────────────────────────────────
   Walls & Floor Generation — Frontend
   ───────────────────────────────────────────────────────────── */

// ── Constants ─────────────────────────────────────────────────
const SCALE = 60;          // pixels per metre on canvas 1
const DEFAULT_WALL_H = 7;  // metres
const GRID_SIZE = SCALE;   // 1 grid square = 1m
const SNAP_RADIUS = 12;    // px — snaps to existing points

const PRODUCT_COLORS = [
  '#E63946','#F4A261','#2A9D8F','#457B9D','#A8DADC',
  '#E9C46A','#E76F51','#6A4C93','#1982C4','#8AC926',
  '#FF595E','#FFCA3A','#C77DFF','#0096C7','#52B788',
  '#D62828','#F77F00','#023E8A','#2D6A4F','#9B2226',
];

// ── State ──────────────────────────────────────────────────────
const S = {
  // Drawing
  tool: 'draw',
  drawPoints: [],        // [{x,y}] in canvas coords (metres * SCALE)
  isClosed: false,
  walls: [],             // [{id,x1,y1,x2,y2,lengthM}]
  hoveredWall: null,
  selectedSurface: null, // {type:'wall'|'floor', wallId?, widthM, heightM}
  mousePos: null,

  // Surface editor
  placedProducts: [],    // [{id,product,color,cx,cy,wPx,hPx}] canvas2 coords
  colorIdx: 0,
  draggingFromSidebar: null,  // product being dragged in
  draggingPlaced: null,       // placed product being moved {id,offX,offY}

  // Generation
  isGenerating: false,
  products: [],
};

let wallIdSeq = 0;
let placedIdSeq = 0;

// ── DOM refs ───────────────────────────────────────────────────
const c1 = document.getElementById('canvas1');
const c2 = document.getElementById('canvas2');
const ctx1 = c1.getContext('2d');
const ctx2 = c2.getContext('2d');

const c1wrap = document.getElementById('canvas1-wrap');
const c2wrap = document.getElementById('canvas2-wrap');
const c1placeholder = document.getElementById('c1-placeholder');
const c2placeholder = document.getElementById('c2-placeholder');
const c2label = document.getElementById('c2-label');
const statusBar = document.getElementById('status-bar');
const catalogList = document.getElementById('catalog-list');
const placedList = document.getElementById('placed-list');
const btnGenerate = document.getElementById('btn-generate');
const noPlacedMsg = document.getElementById('no-placed-msg');
const noSelectionMsg = document.getElementById('no-selection-msg');
const surfaceDetails = document.getElementById('surface-details');
const infoType = document.getElementById('info-type');
const infoWidth = document.getElementById('info-width');
const infoHeight = document.getElementById('info-height');
const resultSection = document.getElementById('result-section');
const resultImg = document.getElementById('result-img');
const resultDownload = document.getElementById('result-download');

// ── Resize canvases ────────────────────────────────────────────
function resizeCanvases() {
  const r1 = c1wrap.getBoundingClientRect();
  const r2 = c2wrap.getBoundingClientRect();
  const hdr = 28; // panel header height
  c1.width  = Math.floor(r1.width);
  c1.height = Math.floor(r1.height) - hdr;
  c2.width  = Math.floor(r2.width);
  c2.height = Math.floor(r2.height) - hdr;
  drawCanvas1();
  renderSurface();
}

const ro = new ResizeObserver(resizeCanvases);
ro.observe(c1wrap);
ro.observe(c2wrap);

// ── Utilities ──────────────────────────────────────────────────
function ptDist(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y);
}
function ptToM(pt) {
  return { x: pt.x / SCALE, y: pt.y / SCALE };
}
function segLengthM(x1, y1, x2, y2) {
  return Math.hypot(x2 - x1, y2 - y1) / SCALE;
}
function snapToGrid(v) {
  return Math.round(v / (GRID_SIZE / 2)) * (GRID_SIZE / 2);
}
function snapToPoints(x, y, exclude) {
  for (const p of S.drawPoints) {
    if (exclude && p === exclude) continue;
    if (ptDist(p, { x, y }) < SNAP_RADIUS) return { ...p, snapped: true };
  }
  return { x: snapToGrid(x), y: snapToGrid(y), snapped: false };
}
function canvasPos(canvas, e) {
  const r = canvas.getBoundingClientRect();
  return { x: e.clientX - r.left, y: e.clientY - r.top };
}
function pickColor() {
  const c = PRODUCT_COLORS[S.colorIdx % PRODUCT_COLORS.length];
  S.colorIdx++;
  return c;
}
function setStatus(msg, cls = '') {
  statusBar.textContent = msg;
  statusBar.className = cls;
}

// ── Canvas 1 drawing ───────────────────────────────────────────
function drawCanvas1() {
  const W = c1.width, H = c1.height;
  ctx1.clearRect(0, 0, W, H);

  // Grid
  ctx1.strokeStyle = '#e8e4de';
  ctx1.lineWidth = 1;
  for (let x = 0; x < W; x += GRID_SIZE) {
    ctx1.beginPath(); ctx1.moveTo(x, 0); ctx1.lineTo(x, H); ctx1.stroke();
  }
  for (let y = 0; y < H; y += GRID_SIZE) {
    ctx1.beginPath(); ctx1.moveTo(0, y); ctx1.lineTo(W, y); ctx1.stroke();
  }
  // Tick labels
  ctx1.fillStyle = '#c0bbb4';
  ctx1.font = '9px sans-serif';
  for (let x = GRID_SIZE; x < W; x += GRID_SIZE) {
    ctx1.fillText(`${x / SCALE}m`, x + 2, 10);
  }
  for (let y = GRID_SIZE; y < H; y += GRID_SIZE) {
    ctx1.fillText(`${y / SCALE}m`, 2, y - 2);
  }

  // Floor fill when closed
  if (S.isClosed && S.drawPoints.length >= 3) {
    const isSelected = S.selectedSurface && S.selectedSurface.type === 'floor';
    ctx1.beginPath();
    ctx1.moveTo(S.drawPoints[0].x, S.drawPoints[0].y);
    S.drawPoints.forEach(p => ctx1.lineTo(p.x, p.y));
    ctx1.closePath();
    ctx1.fillStyle = isSelected ? 'rgba(74,124,89,0.12)' : 'rgba(200,190,178,0.18)';
    ctx1.fill();
    ctx1.strokeStyle = isSelected ? '#4a7c59' : '#9a8a78';
    ctx1.lineWidth = isSelected ? 2.5 : 2;
    ctx1.stroke();
  }

  // Walls (committed segments)
  S.walls.forEach(w => {
    const isHov = S.hoveredWall && S.hoveredWall.id === w.id;
    const isSel = S.selectedSurface && S.selectedSurface.type === 'wall' && S.selectedSurface.wallId === w.id;
    ctx1.beginPath();
    ctx1.moveTo(w.x1, w.y1);
    ctx1.lineTo(w.x2, w.y2);
    ctx1.strokeStyle = isSel ? '#4a7c59' : isHov ? '#7a6a50' : '#5a5048';
    ctx1.lineWidth = isSel ? 4 : isHov ? 3 : 2.5;
    ctx1.lineCap = 'round';
    ctx1.stroke();

    // Length label
    const mx = (w.x1 + w.x2) / 2;
    const my = (w.y1 + w.y2) / 2;
    const dx = w.x2 - w.x1, dy = w.y2 - w.y1;
    const len = Math.hypot(dx, dy);
    const nx = -dy / len * 12, ny = dx / len * 12;

    ctx1.save();
    ctx1.translate(mx + nx, my + ny);
    ctx1.rotate(Math.atan2(dy, dx));
    ctx1.fillStyle = isSel ? '#4a7c59' : '#706050';
    ctx1.font = `bold 10px sans-serif`;
    ctx1.textAlign = 'center';
    ctx1.textBaseline = 'middle';
    ctx1.fillText(`${w.lengthM.toFixed(2)}m`, 0, 0);
    ctx1.restore();
  });

  // In-progress line
  if (S.drawPoints.length > 0 && S.mousePos && !S.isClosed) {
    const last = S.drawPoints[S.drawPoints.length - 1];
    ctx1.beginPath();
    ctx1.moveTo(last.x, last.y);
    ctx1.lineTo(S.mousePos.x, S.mousePos.y);
    ctx1.strokeStyle = '#4a7c59';
    ctx1.lineWidth = 1.5;
    ctx1.setLineDash([5, 4]);
    ctx1.stroke();
    ctx1.setLineDash([]);

    // Preview length
    const lm = segLengthM(last.x, last.y, S.mousePos.x, S.mousePos.y);
    if (lm > 0.05) {
      const mx2 = (last.x + S.mousePos.x) / 2;
      const my2 = (last.y + S.mousePos.y) / 2;
      ctx1.fillStyle = '#4a7c59';
      ctx1.font = '10px sans-serif';
      ctx1.textAlign = 'center';
      ctx1.fillText(`${lm.toFixed(2)}m`, mx2, my2 - 8);
    }
  }

  // Points (corners)
  S.drawPoints.forEach((p, i) => {
    ctx1.beginPath();
    ctx1.arc(p.x, p.y, i === 0 && S.drawPoints.length > 2 && !S.isClosed ? 7 : 4, 0, Math.PI * 2);
    ctx1.fillStyle = i === 0 && S.drawPoints.length > 2 && !S.isClosed ? '#4a7c59' : '#5a5048';
    ctx1.fill();
    if (i === 0 && S.drawPoints.length > 2 && !S.isClosed) {
      ctx1.strokeStyle = '#fff';
      ctx1.lineWidth = 1.5;
      ctx1.stroke();
    }
  });

  // Snap indicator
  if (S.mousePos && S.mousePos.snapped) {
    ctx1.beginPath();
    ctx1.arc(S.mousePos.x, S.mousePos.y, 6, 0, Math.PI * 2);
    ctx1.strokeStyle = '#4a7c59';
    ctx1.lineWidth = 1.5;
    ctx1.stroke();
  }

  c1placeholder.style.display = S.drawPoints.length === 0 ? 'flex' : 'none';
}

// ── Canvas 1 events ────────────────────────────────────────────
c1.addEventListener('mousemove', e => {
  const { x, y } = canvasPos(c1, e);

  if (S.tool === 'draw' && !S.isClosed) {
    S.mousePos = snapToPoints(x, y, S.drawPoints[S.drawPoints.length - 1]);
    drawCanvas1();
    return;
  }

  if (S.tool === 'select' && S.isClosed) {
    S.hoveredWall = hitTestWall(x, y);
    drawCanvas1();
  }
});

c1.addEventListener('mouseleave', () => {
  S.mousePos = null;
  S.hoveredWall = null;
  drawCanvas1();
});

c1.addEventListener('click', e => {
  const { x, y } = canvasPos(c1, e);

  if (S.tool === 'select') {
    const wall = hitTestWall(x, y);
    if (wall) {
      selectWall(wall);
    } else if (S.isClosed && pointInPolygon(x, y, S.drawPoints)) {
      selectFloor();
    }
    return;
  }

  // Draw tool
  if (S.isClosed) return;

  const snapped = snapToPoints(x, y, null);
  // Close polygon if clicking near first point
  if (S.drawPoints.length > 2 && ptDist(snapped, S.drawPoints[0]) < SNAP_RADIUS) {
    closePolygon();
    return;
  }
  addDrawPoint(snapped.x, snapped.y);
});

c1.addEventListener('dblclick', e => {
  if (S.tool !== 'draw' || S.drawPoints.length < 3 || S.isClosed) return;
  closePolygon();
});

function addDrawPoint(x, y) {
  if (S.drawPoints.length > 0) {
    const prev = S.drawPoints[S.drawPoints.length - 1];
    const lm = segLengthM(prev.x, prev.y, x, y);
    if (lm < 0.05) return; // ignore tiny segments
    S.walls.push({ id: ++wallIdSeq, x1: prev.x, y1: prev.y, x2: x, y2: y, lengthM: lm });
  }
  S.drawPoints.push({ x, y });
  c1placeholder.style.display = 'none';
  drawCanvas1();
}

function closePolygon() {
  if (S.drawPoints.length < 3) return;
  const first = S.drawPoints[0];
  const last = S.drawPoints[S.drawPoints.length - 1];
  const lm = segLengthM(last.x, last.y, first.x, first.y);
  if (lm > 0.05) {
    S.walls.push({ id: ++wallIdSeq, x1: last.x, y1: last.y, x2: first.x, y2: first.y, lengthM: lm });
  }
  S.isClosed = true;
  S.mousePos = null;
  setStatus('Room closed. Click a wall or floor to select it.', '');
  document.getElementById('c1-hint').textContent = 'Click wall to edit · Click floor area for floor view';
  drawCanvas1();
}

function hitTestWall(x, y, thresh = 8) {
  for (const w of S.walls) {
    const dx = w.x2 - w.x1, dy = w.y2 - w.y1;
    const len2 = dx * dx + dy * dy;
    if (len2 === 0) continue;
    const t = Math.max(0, Math.min(1, ((x - w.x1) * dx + (y - w.y1) * dy) / len2));
    const px = w.x1 + t * dx, py = w.y1 + t * dy;
    if (Math.hypot(x - px, y - py) < thresh) return w;
  }
  return null;
}

function pointInPolygon(x, y, pts) {
  let inside = false;
  for (let i = 0, j = pts.length - 1; i < pts.length; j = i++) {
    const xi = pts[i].x, yi = pts[i].y;
    const xj = pts[j].x, yj = pts[j].y;
    if (((yi > y) !== (yj > y)) && (x < (xj - xi) * (y - yi) / (yj - yi) + xi)) {
      inside = !inside;
    }
  }
  return inside;
}

function selectWall(wall) {
  S.selectedSurface = {
    type: 'wall',
    wallId: wall.id,
    widthM: wall.lengthM,
    heightM: DEFAULT_WALL_H,
    wall,
  };
  S.placedProducts = [];
  S.colorIdx = 0;
  updateSurfaceInfo();
  drawCanvas1();
  renderSurface();
  updatePlacedList();
  updateGenerateBtn();
}

function selectFloor() {
  // Compute bounding box of the floor polygon
  const xs = S.drawPoints.map(p => p.x);
  const ys = S.drawPoints.map(p => p.y);
  const wM = (Math.max(...xs) - Math.min(...xs)) / SCALE;
  const hM = (Math.max(...ys) - Math.min(...ys)) / SCALE;
  S.selectedSurface = {
    type: 'floor',
    widthM: wM,
    heightM: hM,
    polygon: S.drawPoints,
  };
  S.placedProducts = [];
  S.colorIdx = 0;
  updateSurfaceInfo();
  drawCanvas1();
  renderSurface();
  updatePlacedList();
  updateGenerateBtn();
}

// ── Surface info ───────────────────────────────────────────────
function updateSurfaceInfo() {
  if (!S.selectedSurface) {
    noSelectionMsg.style.display = '';
    surfaceDetails.style.display = 'none';
    c2label.textContent = 'Select a wall or floor in Canvas 1';
    return;
  }
  noSelectionMsg.style.display = 'none';
  surfaceDetails.style.display = '';
  infoType.textContent = S.selectedSurface.type === 'wall' ? 'Wall' : 'Floor';
  infoWidth.textContent = `${S.selectedSurface.widthM.toFixed(2)} m`;
  infoHeight.textContent = `${S.selectedSurface.heightM.toFixed(2)} m`;

  if (S.selectedSurface.type === 'wall') {
    const w = S.selectedSurface.wall;
    const wIdx = S.walls.findIndex(x => x.id === w.id) + 1;
    c2label.textContent = `Wall ${wIdx} · ${w.lengthM.toFixed(2)}m × ${DEFAULT_WALL_H}m`;
  } else {
    c2label.textContent = `Floor · ${S.selectedSurface.widthM.toFixed(2)}m × ${S.selectedSurface.heightM.toFixed(2)}m`;
  }
}

// ── Canvas 2: render surface ───────────────────────────────────
function renderSurface() {
  const W = c2.width, H = c2.height;
  ctx2.clearRect(0, 0, W, H);
  c2placeholder.style.display = 'none';

  if (!S.selectedSurface) {
    c2placeholder.style.display = 'flex';
    return;
  }

  const { widthM, heightM } = S.selectedSurface;
  const PAD = 24;
  const scaleX = (W - PAD * 2) / widthM;
  const scaleY = (H - PAD * 2) / heightM;
  const sc = Math.min(scaleX, scaleY);
  const surfW = widthM * sc;
  const surfH = heightM * sc;
  const ox = (W - surfW) / 2;
  const oy = (H - surfH) / 2;

  // Store transform for hit testing
  c2._transform = { ox, oy, sc, surfW, surfH, widthM, heightM };

  // Surface background
  if (S.selectedSurface.type === 'wall') {
    ctx2.fillStyle = '#f5f2ee';
    ctx2.fillRect(ox, oy, surfW, surfH);
    // subtle brick / plaster texture lines
    ctx2.strokeStyle = '#e8e0d8';
    ctx2.lineWidth = 0.5;
    for (let y = oy + 30; y < oy + surfH; y += 30) {
      ctx2.beginPath(); ctx2.moveTo(ox, y); ctx2.lineTo(ox + surfW, y); ctx2.stroke();
    }
  } else {
    // Floor: light tile pattern
    ctx2.fillStyle = '#eeeae4';
    ctx2.fillRect(ox, oy, surfW, surfH);
    ctx2.strokeStyle = '#e0dbd4';
    ctx2.lineWidth = 0.5;
    const tile = 60;
    for (let x = ox; x < ox + surfW; x += tile) {
      ctx2.beginPath(); ctx2.moveTo(x, oy); ctx2.lineTo(x, oy + surfH); ctx2.stroke();
    }
    for (let y = oy; y < oy + surfH; y += tile) {
      ctx2.beginPath(); ctx2.moveTo(ox, y); ctx2.lineTo(ox + surfW, y); ctx2.stroke();
    }
  }

  // Border
  ctx2.strokeStyle = '#7a7068';
  ctx2.lineWidth = 1.5;
  ctx2.strokeRect(ox, oy, surfW, surfH);

  // Floor/ceiling lines for wall view
  if (S.selectedSurface.type === 'wall') {
    ctx2.fillStyle = '#c8c0b0';
    ctx2.fillRect(ox, oy + surfH, surfW, 6);
    ctx2.fillStyle = '#e0d8d0';
    ctx2.fillRect(ox, oy - 4, surfW, 4);
  }

  // Placed products
  S.placedProducts.forEach(pp => {
    ctx2.fillStyle = pp.color + 'cc';
    ctx2.fillRect(pp.cx, pp.cy, pp.wPx, pp.hPx);
    ctx2.strokeStyle = pp.color;
    ctx2.lineWidth = 2;
    ctx2.strokeRect(pp.cx, pp.cy, pp.wPx, pp.hPx);

    // Product icon inside
    if (pp.img) {
      const iconPad = 4;
      ctx2.drawImage(pp.img, pp.cx + iconPad, pp.cy + iconPad, pp.wPx - iconPad * 2, pp.hPx - iconPad * 2);
    }

    // Label
    ctx2.fillStyle = '#fff';
    ctx2.font = 'bold 9px sans-serif';
    ctx2.textAlign = 'center';
    ctx2.fillText(pp.product.name, pp.cx + pp.wPx / 2, pp.cy + pp.hPx / 2);
  });

  // Dimension labels
  ctx2.fillStyle = '#9a9088';
  ctx2.font = '10px sans-serif';
  ctx2.textAlign = 'center';
  ctx2.fillText(`${widthM.toFixed(2)} m`, ox + surfW / 2, oy + surfH + 18);
  ctx2.save();
  ctx2.translate(ox - 14, oy + surfH / 2);
  ctx2.rotate(-Math.PI / 2);
  ctx2.fillText(`${heightM.toFixed(2)} m`, 0, 0);
  ctx2.restore();
}

// ── Canvas 2: drag/drop from sidebar ──────────────────────────
c2.addEventListener('dragover', e => {
  e.preventDefault();
  c2.classList.add('drag-over');
});
c2.addEventListener('dragleave', () => c2.classList.remove('drag-over'));

c2.addEventListener('drop', e => {
  e.preventDefault();
  c2.classList.remove('drag-over');
  if (!S.selectedSurface || !S.draggingFromSidebar) return;

  const { x, y } = canvasPos(c2, e);
  placeProduct(S.draggingFromSidebar, x, y);
  S.draggingFromSidebar = null;
});

function placeProduct(product, dropX, dropY) {
  const t = c2._transform;
  if (!t) return;

  // Product pixel dimensions on canvas2
  const dims = product.dimensions || {};
  const wM = dims.width || 1;
  const dM = dims.depth || dims.height || wM;
  const wPx = Math.max(30, wM * t.sc);
  const hPx = Math.max(30, dM * t.sc);

  // Center drop point, clamp inside surface
  const cx = Math.max(t.ox, Math.min(t.ox + t.surfW - wPx, dropX - wPx / 2));
  const cy = Math.max(t.oy, Math.min(t.oy + t.surfH - hPx, dropY - hPx / 2));

  const color = pickColor();

  // Preload icon image
  const img = new Image();
  img.src = product.icon;

  const pp = { id: ++placedIdSeq, product, color, cx, cy, wPx, hPx, img };
  img.onload = () => renderSurface();

  S.placedProducts.push(pp);
  updatePlacedList();
  updateGenerateBtn();
  renderSurface();
}

// Move placed products
c2.addEventListener('mousedown', e => {
  const { x, y } = canvasPos(c2, e);
  for (let i = S.placedProducts.length - 1; i >= 0; i--) {
    const pp = S.placedProducts[i];
    if (x >= pp.cx && x <= pp.cx + pp.wPx && y >= pp.cy && y <= pp.cy + pp.hPx) {
      S.draggingPlaced = { id: pp.id, offX: x - pp.cx, offY: y - pp.cy };
      c2.style.cursor = 'grabbing';
      return;
    }
  }
});

c2.addEventListener('mousemove', e => {
  if (!S.draggingPlaced) return;
  const { x, y } = canvasPos(c2, e);
  const pp = S.placedProducts.find(p => p.id === S.draggingPlaced.id);
  if (!pp) return;
  const t = c2._transform;
  if (!t) return;
  pp.cx = Math.max(t.ox, Math.min(t.ox + t.surfW - pp.wPx, x - S.draggingPlaced.offX));
  pp.cy = Math.max(t.oy, Math.min(t.oy + t.surfH - pp.hPx, y - S.draggingPlaced.offY));
  renderSurface();
});

c2.addEventListener('mouseup', () => {
  S.draggingPlaced = null;
  c2.style.cursor = 'default';
});

// ── Placed products list ───────────────────────────────────────
function updatePlacedList() {
  if (S.placedProducts.length === 0) {
    placedList.innerHTML = '';
    placedList.appendChild(noPlacedMsg);
    noPlacedMsg.style.display = '';
    return;
  }
  noPlacedMsg.style.display = 'none';
  const items = S.placedProducts.map(pp => {
    const div = document.createElement('div');
    div.className = 'placed-item';
    div.innerHTML = `
      <div class="color-swatch" style="background:${pp.color}"></div>
      <span class="name">${pp.product.name}</span>
      <button class="remove-btn" data-id="${pp.id}" title="Remove">×</button>
    `;
    return div;
  });
  placedList.innerHTML = '';
  items.forEach(el => {
    el.querySelector('.remove-btn').addEventListener('click', evt => {
      const id = parseInt(evt.target.dataset.id);
      S.placedProducts = S.placedProducts.filter(p => p.id !== id);
      updatePlacedList();
      updateGenerateBtn();
      renderSurface();
    });
    placedList.appendChild(el);
  });
}

function updateGenerateBtn() {
  btnGenerate.disabled = S.placedProducts.length === 0 || S.isGenerating;
}

// ── Toolbar buttons ────────────────────────────────────────────
document.getElementById('tool-draw').addEventListener('click', () => setTool('draw'));
document.getElementById('tool-select').addEventListener('click', () => setTool('select'));

function setTool(t) {
  S.tool = t;
  document.querySelectorAll('.tool-btn').forEach(b => b.classList.remove('active'));
  document.getElementById('tool-' + t).classList.add('active');
  c1.style.cursor = t === 'draw' ? 'crosshair' : 'default';
}

document.getElementById('tool-undo').addEventListener('click', () => {
  if (S.isClosed || S.drawPoints.length === 0) return;
  S.drawPoints.pop();
  S.walls.pop();
  drawCanvas1();
});

document.getElementById('tool-clear').addEventListener('click', () => {
  S.drawPoints = [];
  S.walls = [];
  S.isClosed = false;
  S.selectedSurface = null;
  S.placedProducts = [];
  S.colorIdx = 0;
  wallIdSeq = 0;
  placedIdSeq = 0;
  document.getElementById('c1-hint').textContent = 'Click to draw walls · Double-click to close room';
  updateSurfaceInfo();
  updatePlacedList();
  updateGenerateBtn();
  renderSurface();
  drawCanvas1();
  setStatus('Ready', '');
});

document.getElementById('tool-close').addEventListener('click', () => {
  if (!S.isClosed && S.drawPoints.length >= 3) closePolygon();
});

// ── Catalog / sidebar ──────────────────────────────────────────
async function loadCatalog() {
  try {
    const res = await fetch('/api/products');
    S.products = await res.json();
    renderCatalog();
  } catch {
    catalogList.innerHTML = '<div style="font-size:12px;color:var(--danger);padding:8px">Failed to load products</div>';
  }
}

function renderCatalog() {
  catalogList.innerHTML = '';
  S.products.forEach(product => {
    const div = document.createElement('div');
    div.className = 'catalog-item';
    div.draggable = true;
    div.innerHTML = `
      <div class="icon-wrap">
        <img src="${product.icon}" alt="${product.name}" loading="lazy" />
      </div>
      <div class="info">
        <div class="name">${product.name}</div>
        <div class="dims">${product.dims || ''}</div>
      </div>
    `;
    div.addEventListener('dragstart', e => {
      S.draggingFromSidebar = product;
      e.dataTransfer.effectAllowed = 'copy';
    });
    div.addEventListener('dragend', () => { S.draggingFromSidebar = null; });
    catalogList.appendChild(div);
  });
}

// ── Export highlight image ─────────────────────────────────────
function exportHighlightImage() {
  // Create offscreen canvas with only the colored zones (no icons/labels)
  const oc = document.createElement('canvas');
  oc.width = 1024;
  oc.height = 1024;
  const oc2d = oc.getContext('2d');

  if (!S.selectedSurface || !c2._transform) return null;
  const t = c2._transform;

  // Scale factor from canvas2 space to 1024x1024
  const sx = 1024 / c2.width;
  const sy = 1024 / c2.height;

  // White background (the wall/floor surface)
  oc2d.fillStyle = '#ffffff';
  oc2d.fillRect(
    t.ox * sx, t.oy * sy,
    t.surfW * sx, t.surfH * sy
  );

  // Colored product zones
  S.placedProducts.forEach(pp => {
    oc2d.fillStyle = pp.color;
    oc2d.fillRect(pp.cx * sx, pp.cy * sy, pp.wPx * sx, pp.hPx * sy);
  });

  return oc.toDataURL('image/png');
}

// ── Generate ───────────────────────────────────────────────────
btnGenerate.addEventListener('click', async () => {
  if (S.isGenerating || S.placedProducts.length === 0 || !S.selectedSurface) return;

  const highlightImg = exportHighlightImage();
  if (!highlightImg) { setStatus('Cannot export surface image', 'error'); return; }

  const t = c2._transform;
  const products = S.placedProducts.map(pp => ({
    product_id: pp.product.id,
    image_url: `${window.location.origin}${pp.product.image_url}`,
    hex_color: pp.color,
    dims: pp.product.dims || '',
    rotation: 0,
  }));

  const roomDimensions = {
    width: S.selectedSurface.widthM,
    height: S.selectedSurface.heightM,
    unit: 'm',
  };

  const body = {
    highlight_image: highlightImg,
    products,
    presets: {},
    room_dimensions: roomDimensions,
    type: S.selectedSurface.type,
  };

  S.isGenerating = true;
  btnGenerate.disabled = true;
  setStatus('Starting generation…', 'processing');
  resultSection.style.display = 'none';

  let genId;
  try {
    const res = await fetch('/api/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();
    genId = data.gen_id;
    setStatus('Generating… this may take a minute', 'processing');
  } catch (err) {
    setStatus(`Failed to start: ${err.message}`, 'error');
    S.isGenerating = false;
    updateGenerateBtn();
    return;
  }

  // Poll
  pollGeneration(genId);
});

async function pollGeneration(genId) {
  const INTERVAL = 2000;
  const MAX_WAIT = 5 * 60 * 1000; // 5 min
  const start = Date.now();

  const poll = async () => {
    if (Date.now() - start > MAX_WAIT) {
      setStatus('Generation timed out', 'error');
      S.isGenerating = false;
      updateGenerateBtn();
      return;
    }

    try {
      const res = await fetch(`/api/generate/${genId}`);
      const data = await res.json();

      if (data.status === 'success') {
        setStatus('Generation complete!', 'success');
        S.isGenerating = false;
        updateGenerateBtn();
        showResult(data.result_image);
      } else if (data.status === 'failed') {
        setStatus(`Generation failed: ${data.reason || 'unknown error'}`, 'error');
        S.isGenerating = false;
        updateGenerateBtn();
      } else {
        setTimeout(poll, INTERVAL);
      }
    } catch {
      setTimeout(poll, INTERVAL);
    }
  };

  setTimeout(poll, INTERVAL);
}

function showResult(imageUrl) {
  resultSection.style.display = '';
  resultImg.src = imageUrl;
  resultImg.classList.add('visible');
  resultDownload.href = imageUrl;
  resultDownload.download = 'generated.png';
  resultDownload.style.display = 'block';
}

// ── Init ───────────────────────────────────────────────────────
loadCatalog();
resizeCanvases();
