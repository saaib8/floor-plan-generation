/* ─────────────────────────────────────────────────────────────
   Walls & Floor Generation — Frontend
   ───────────────────────────────────────────────────────────── */

// ── Constants ─────────────────────────────────────────────────
const SCALE = 60;          // pixels per metre on canvas 1
const DEFAULT_WALL_H = 2.8;  // metres
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

  // Openings
  openings: [],           // [{id, wallId, type, posAlongWall, widthM}]
  addingOpening: null,    // null | 'door' | 'window'

  // Generation
  isGenerating: false,
  products: [],
};

let wallIdSeq = 0;
let placedIdSeq = 0;
let openingIdSeq = 0;

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
const openingsSection = document.getElementById('openings-section');
const openingsList = document.getElementById('openings-list');
const noOpeningsMsg = document.getElementById('no-openings-msg');
const btnAddDoor = document.getElementById('btn-add-door');
const btnAddWindow = document.getElementById('btn-add-window');

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

  // Openings on walls
  S.openings.forEach(op => {
    const w = S.walls.find(wl => wl.id === op.wallId);
    if (!w) return;
    const dx = w.x2 - w.x1, dy = w.y2 - w.y1;
    const wallLen = Math.hypot(dx, dy);
    if (wallLen === 0) return;
    const ux = dx / wallLen, uy = dy / wallLen; // unit along wall
    const nx = -uy, ny = ux; // normal (perpendicular)
    const widthPx = op.widthM * SCALE;
    const centerPx = op.posAlongWall * wallLen;
    const halfW = widthPx / 2;
    const startT = Math.max(0, centerPx - halfW);
    const endT = Math.min(wallLen, centerPx + halfW);
    const sx = w.x1 + ux * startT, sy = w.y1 + uy * startT;
    const ex = w.x1 + ux * endT, ey = w.y1 + uy * endT;
    const thick = 6; // half-thickness of the gap overlay
    const color = op.type === 'door' ? '#F0E4D2' : '#B4D7F0';
    ctx1.save();
    ctx1.beginPath();
    ctx1.moveTo(sx + nx * thick, sy + ny * thick);
    ctx1.lineTo(ex + nx * thick, ey + ny * thick);
    ctx1.lineTo(ex - nx * thick, ey - ny * thick);
    ctx1.lineTo(sx - nx * thick, sy - ny * thick);
    ctx1.closePath();
    ctx1.fillStyle = color;
    ctx1.fill();
    ctx1.strokeStyle = op.type === 'door' ? '#c8a878' : '#78a8c8';
    ctx1.lineWidth = 1;
    ctx1.stroke();
    // Label
    const mx = (sx + ex) / 2, my = (sy + ey) / 2;
    ctx1.font = '9px sans-serif';
    ctx1.textAlign = 'center';
    ctx1.textBaseline = 'middle';
    ctx1.fillStyle = '#5a5048';
    ctx1.fillText(`${op.type === 'door' ? 'D' : 'W'} ${op.widthM.toFixed(1)}m`, mx + nx * 14, my + ny * 14);
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

  // Opening placement mode
  if (S.addingOpening && S.isClosed) {
    const wall = hitTestWall(x, y, 12);
    if (wall) {
      const dx = wall.x2 - wall.x1, dy = wall.y2 - wall.y1;
      const len2 = dx * dx + dy * dy;
      const t = Math.max(0, Math.min(1, ((x - wall.x1) * dx + (y - wall.y1) * dy) / len2));
      placeOpening(wall.id, t, S.addingOpening);
      cancelAddOpening();
    } else {
      setStatus('Click on a wall segment to place the opening', '');
    }
    return;
  }

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
  showOpeningsSection();
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

  // Openings on floor surface edges
  if (S.selectedSurface.type === 'floor' && S.openings.length > 0) {
    const xs = S.drawPoints.map(p => p.x);
    const ys = S.drawPoints.map(p => p.y);
    const minX = Math.min(...xs), maxX = Math.max(...xs);
    const minY = Math.min(...ys), maxY = Math.max(...ys);
    const roomWPx = maxX - minX, roomHPx = maxY - minY;
    const thick = 5; // visual thickness of the opening marker (px)

    S.openings.forEach(op => {
      const wall = S.walls.find(w => w.id === op.wallId);
      if (!wall) return;
      const compass = wallToCompass(wall);
      // Compute opening center position along wall in canvas1 coords
      const dx = wall.x2 - wall.x1, dy = wall.y2 - wall.y1;
      const wallLen = Math.hypot(dx, dy);
      const centerAlongWall = op.posAlongWall * wallLen;
      const ptX = wall.x1 + (dx / wallLen) * centerAlongWall;
      const ptY = wall.y1 + (dy / wallLen) * centerAlongWall;
      const halfWPx = (op.widthM * SCALE) / 2;

      const color = op.type === 'door' ? '#E8D4B0' : '#A8CCE8';
      const borderColor = op.type === 'door' ? '#C0A070' : '#6898C0';
      const label = op.type === 'door' ? 'D' : 'W';

      let rx, ry, rw, rh;
      if (compass === 'north') {
        const fracL = ((ptX - halfWPx) - minX) / roomWPx;
        const fracR = ((ptX + halfWPx) - minX) / roomWPx;
        rx = ox + fracL * surfW;
        rw = (fracR - fracL) * surfW;
        ry = oy - thick;
        rh = thick * 2;
      } else if (compass === 'south') {
        const fracL = ((ptX - halfWPx) - minX) / roomWPx;
        const fracR = ((ptX + halfWPx) - minX) / roomWPx;
        rx = ox + fracL * surfW;
        rw = (fracR - fracL) * surfW;
        ry = oy + surfH - thick;
        rh = thick * 2;
      } else if (compass === 'west') {
        const fracT = ((ptY - halfWPx) - minY) / roomHPx;
        const fracB = ((ptY + halfWPx) - minY) / roomHPx;
        rx = ox - thick;
        rw = thick * 2;
        ry = oy + fracT * surfH;
        rh = (fracB - fracT) * surfH;
      } else { // east
        const fracT = ((ptY - halfWPx) - minY) / roomHPx;
        const fracB = ((ptY + halfWPx) - minY) / roomHPx;
        rx = ox + surfW - thick;
        rw = thick * 2;
        ry = oy + fracT * surfH;
        rh = (fracB - fracT) * surfH;
      }
      ctx2.fillStyle = color;
      ctx2.fillRect(rx, ry, rw, rh);
      ctx2.strokeStyle = borderColor;
      ctx2.lineWidth = 1;
      ctx2.strokeRect(rx, ry, rw, rh);
      // Label
      ctx2.fillStyle = '#5a5048';
      ctx2.font = 'bold 9px sans-serif';
      ctx2.textAlign = 'center';
      ctx2.textBaseline = 'middle';
      ctx2.fillText(label, rx + rw / 2, ry + rh / 2);
    });
  }

  // Floor/ceiling lines for wall view
  if (S.selectedSurface.type === 'wall') {
    ctx2.fillStyle = '#c8c0b0';
    ctx2.fillRect(ox, oy + surfH, surfW, 6);
    ctx2.fillStyle = '#e0d8d0';
    ctx2.fillRect(ox, oy - 4, surfW, 4);
  }

  // Placed products
  S.placedProducts.forEach(pp => {
    const rot = (pp.rotation || 0) * Math.PI / 180;
    const cxMid = pp.cx + pp.wPx / 2;
    const cyMid = pp.cy + pp.hPx / 2;

    // Colored fill rect (dimensions already reflect rotation via wPx/hPx swap)
    ctx2.fillStyle = pp.color + 'cc';
    ctx2.fillRect(pp.cx, pp.cy, pp.wPx, pp.hPx);
    ctx2.strokeStyle = pp.color;
    ctx2.lineWidth = 2;
    ctx2.strokeRect(pp.cx, pp.cy, pp.wPx, pp.hPx);

    // Product icon — draw rotated so orientation is visually correct
    if (pp.img) {
      ctx2.save();
      ctx2.translate(cxMid, cyMid);
      ctx2.rotate(rot);
      // In rotated context, natural image dims are the pre-swap dimensions
      const imgW = (pp.rotation % 180 === 0) ? pp.wPx : pp.hPx;
      const imgH = (pp.rotation % 180 === 0) ? pp.hPx : pp.wPx;
      const pad = 4;
      ctx2.drawImage(pp.img, -(imgW / 2) + pad, -(imgH / 2) + pad, imgW - pad * 2, imgH - pad * 2);
      ctx2.restore();
    }

    // Label + rotation badge
    ctx2.fillStyle = '#fff';
    ctx2.font = 'bold 9px sans-serif';
    ctx2.textAlign = 'center';
    ctx2.textBaseline = 'middle';
    ctx2.fillText(pp.product.name, cxMid, cyMid);
    if (pp.rotation) {
      ctx2.font = '8px sans-serif';
      ctx2.fillStyle = 'rgba(0,0,0,0.6)';
      ctx2.fillRect(pp.cx + pp.wPx - 22, pp.cy + 2, 20, 12);
      ctx2.fillStyle = '#fff';
      ctx2.textAlign = 'center';
      ctx2.fillText(`${pp.rotation}°`, pp.cx + pp.wPx - 12, pp.cy + 9);
    }
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

  // Preload icon image (for canvas display only)
  const img = new Image();
  img.src = product.icon;

  // imageUrl: the real product photo URL used for AI generation (not the SVG icon)
  const pp = { id: ++placedIdSeq, product, color, cx, cy, wPx, hPx, img, imageUrl: '', rotation: 0 };
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
    const hasUrl = !!pp.imageUrl.trim();
    div.innerHTML = `
      <div class="placed-item-row">
        <div class="color-swatch" style="background:${pp.color}"></div>
        <span class="name">${pp.product.name}</span>
        <button class="rotate-btn" data-id="${pp.id}" title="Rotate 90°">↻ ${pp.rotation || 0}°</button>
        <button class="remove-btn" data-id="${pp.id}" title="Remove">×</button>
      </div>
      <input
        class="placed-img-url ${hasUrl ? 'has-url' : ''}"
        data-id="${pp.id}"
        type="url"
        placeholder="Paste product photo URL (PNG/JPG)"
        value="${pp.imageUrl}"
      />
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
    el.querySelector('.rotate-btn').addEventListener('click', evt => {
      const id = parseInt(evt.target.dataset.id);
      const pp = S.placedProducts.find(p => p.id === id);
      if (!pp) return;
      const wasPortrait = pp.rotation % 180 !== 0;
      pp.rotation = (pp.rotation + 90) % 360;
      const isPortrait = pp.rotation % 180 !== 0;
      // Swap wPx/hPx when toggling between landscape (0°/180°) and portrait (90°/270°)
      if (wasPortrait !== isPortrait) {
        [pp.wPx, pp.hPx] = [pp.hPx, pp.wPx];
        // Re-clamp so product stays inside surface bounds
        const t = c2._transform;
        if (t) {
          pp.cx = Math.max(t.ox, Math.min(t.ox + t.surfW - pp.wPx, pp.cx));
          pp.cy = Math.max(t.oy, Math.min(t.oy + t.surfH - pp.hPx, pp.cy));
        }
      }
      updatePlacedList();
      renderSurface();
    });
    el.querySelector('.placed-img-url').addEventListener('input', evt => {
      const id = parseInt(evt.target.dataset.id);
      const pp = S.placedProducts.find(p => p.id === id);
      if (pp) {
        pp.imageUrl = evt.target.value;
        evt.target.classList.toggle('has-url', !!pp.imageUrl.trim());
      }
      updateGenerateBtn();
    });
    placedList.appendChild(el);
  });
}

function updateGenerateBtn() {
  const hasProducts = S.placedProducts.length > 0;
  const allHaveUrls = S.placedProducts.every(pp => pp.imageUrl.trim());
  btnGenerate.disabled = !hasProducts || !allHaveUrls || S.isGenerating;
  btnGenerate.title = hasProducts && !allHaveUrls
    ? 'Add a product photo URL for each placed product'
    : '';
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
  S.openings = [];
  S.addingOpening = null;
  S.colorIdx = 0;
  wallIdSeq = 0;
  placedIdSeq = 0;
  openingIdSeq = 0;
  document.getElementById('c1-hint').textContent = 'Click to draw walls · Double-click to close room';
  updateSurfaceInfo();
  updatePlacedList();
  updateOpeningsList();
  showOpeningsSection();
  updateGenerateBtn();
  renderSurface();
  drawCanvas1();
  setStatus('Ready', '');
});

document.getElementById('tool-close').addEventListener('click', () => {
  if (!S.isClosed && S.drawPoints.length >= 3) closePolygon();
});

// ── Openings (doors & windows) ─────────────────────────────
function showOpeningsSection() {
  openingsSection.style.display = S.isClosed ? '' : 'none';
}

function startAddOpening(type) {
  if (S.addingOpening === type) { cancelAddOpening(); return; }
  S.addingOpening = type;
  btnAddDoor.classList.toggle('active', type === 'door');
  btnAddWindow.classList.toggle('active', type === 'window');
  c1.style.cursor = 'pointer';
  setStatus(`Click on a wall to place a ${type}`, '');
}

function cancelAddOpening() {
  S.addingOpening = null;
  btnAddDoor.classList.remove('active');
  btnAddWindow.classList.remove('active');
  c1.style.cursor = S.tool === 'draw' ? 'crosshair' : 'default';
  setStatus(S.isClosed ? 'Room closed. Click a wall or floor to select it.' : 'Ready', '');
}

function placeOpening(wallId, posAlongWall, type) {
  const wall = S.walls.find(w => w.id === wallId);
  if (!wall) return;
  const defaultW = type === 'door' ? 0.9 : 1.5;
  // Clamp width so it doesn't exceed wall length
  const widthM = Math.min(defaultW, wall.lengthM * 0.95);
  // Clamp position so opening stays within wall
  const halfFrac = (widthM / wall.lengthM) / 2;
  const clampedPos = Math.max(halfFrac, Math.min(1 - halfFrac, posAlongWall));
  S.openings.push({
    id: ++openingIdSeq,
    wallId,
    type,
    posAlongWall: clampedPos,
    widthM,
  });
  updateOpeningsList();
  drawCanvas1();
}

function updateOpeningsList() {
  if (S.openings.length === 0) {
    openingsList.innerHTML = '';
    openingsList.appendChild(noOpeningsMsg);
    noOpeningsMsg.style.display = '';
    return;
  }
  noOpeningsMsg.style.display = 'none';
  const items = S.openings.map(op => {
    const wallIdx = S.walls.findIndex(w => w.id === op.wallId) + 1;
    const div = document.createElement('div');
    div.className = 'opening-item';
    div.innerHTML = `
      <span class="opening-icon">${op.type === 'door' ? '🚪' : '🪟'}</span>
      <span class="opening-label">Wall ${wallIdx}</span>
      <input class="width-input" type="number" step="0.1" min="0.3" max="10" value="${op.widthM.toFixed(1)}" data-id="${op.id}" title="Width (m)" />
      <span style="font-size:9px;color:var(--text-muted)">m</span>
      <button class="remove-btn" data-id="${op.id}" title="Remove">×</button>
    `;
    return div;
  });
  openingsList.innerHTML = '';
  items.forEach(el => {
    el.querySelector('.remove-btn').addEventListener('click', evt => {
      const id = parseInt(evt.target.dataset.id);
      S.openings = S.openings.filter(o => o.id !== id);
      updateOpeningsList();
      drawCanvas1();
    });
    el.querySelector('.width-input').addEventListener('input', evt => {
      const id = parseInt(evt.target.dataset.id);
      const op = S.openings.find(o => o.id === id);
      if (!op) return;
      const wall = S.walls.find(w => w.id === op.wallId);
      let val = parseFloat(evt.target.value) || 0.3;
      if (wall) val = Math.min(val, wall.lengthM * 0.95);
      op.widthM = Math.max(0.3, val);
      // Re-clamp position
      if (wall) {
        const halfFrac = (op.widthM / wall.lengthM) / 2;
        op.posAlongWall = Math.max(halfFrac, Math.min(1 - halfFrac, op.posAlongWall));
      }
      drawCanvas1();
    });
    openingsList.appendChild(el);
  });
}

function wallToCompass(wall) {
  const dx = wall.x2 - wall.x1;
  const dy = wall.y2 - wall.y1;
  const isHorizontal = Math.abs(dx) >= Math.abs(dy);
  const xs = S.drawPoints.map(p => p.x);
  const ys = S.drawPoints.map(p => p.y);
  const centerY = (Math.min(...ys) + Math.max(...ys)) / 2;
  const centerX = (Math.min(...xs) + Math.max(...xs)) / 2;
  const midY = (wall.y1 + wall.y2) / 2;
  const midX = (wall.x1 + wall.x2) / 2;
  if (isHorizontal) {
    return midY < centerY ? 'north' : 'south';
  } else {
    return midX < centerX ? 'west' : 'east';
  }
}

function mapOpeningsToCompass() {
  const xs = S.drawPoints.map(p => p.x);
  const ys = S.drawPoints.map(p => p.y);
  const minX = Math.min(...xs), maxX = Math.max(...xs);
  const minY = Math.min(...ys), maxY = Math.max(...ys);
  const roomWM = (maxX - minX) / SCALE;
  const roomHM = (maxY - minY) / SCALE;

  return S.openings.map((op, i) => {
    const wall = S.walls.find(w => w.id === op.wallId);
    if (!wall) return null;
    const compass = wallToCompass(wall);
    const dx = wall.x2 - wall.x1, dy = wall.y2 - wall.y1;
    const wallLen = Math.hypot(dx, dy);
    const centerAlongWall = op.posAlongWall * wallLen; // px along wall
    // Point on wall at the opening center
    const ptX = wall.x1 + (dx / wallLen) * centerAlongWall;
    const ptY = wall.y1 + (dy / wallLen) * centerAlongWall;
    // Project onto compass axis to get position_from_left
    let posFromLeft = 0;
    if (compass === 'north' || compass === 'south') {
      posFromLeft = (ptX - minX) / SCALE - op.widthM / 2;
    } else {
      posFromLeft = (ptY - minY) / SCALE - op.widthM / 2;
    }
    posFromLeft = Math.max(0, posFromLeft);

    return {
      id: `${op.type}_${String(i + 1).padStart(2, '0')}`,
      type: op.type,
      wall: compass,
      position_from_left: parseFloat(posFromLeft.toFixed(3)),
      width: op.widthM,
      height: op.type === 'door' ? 2.2 : 0.9,
      sill_height: op.type === 'window' ? 1.2 : undefined,
    };
  }).filter(Boolean);
}

btnAddDoor.addEventListener('click', () => startAddOpening('door'));
btnAddWindow.addEventListener('click', () => startAddOpening('window'));

// Escape key cancels opening placement
document.addEventListener('keydown', e => {
  if (e.key === 'Escape' && S.addingOpening) {
    cancelAddOpening();
  }
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
  const products = S.placedProducts.map(pp => {
    // Center position in metres within the surface
    const x_m = t ? parseFloat(((pp.cx - t.ox + pp.wPx / 2) / t.sc).toFixed(3)) : null;
    const y_m = t ? parseFloat(((pp.cy - t.oy + pp.hPx / 2) / t.sc).toFixed(3)) : null;
    return {
      product_id: pp.product.id,
      category: pp.product.category || '',
      image_url: pp.imageUrl.trim(),
      hex_color: pp.color,
      dims: pp.product.dims || '',
      dimensions: pp.product.dimensions || null,  // physical (unrotated) metres
      rotation: pp.rotation || 0,
      x_m,
      y_m,
    };
  });

  const roomDimensions = {
    width: S.selectedSurface.widthM,
    height: S.selectedSurface.heightM,
    unit: 'm',
  };

  const openings = S.selectedSurface.type === 'floor' ? mapOpeningsToCompass() : [];

  const body = {
    highlight_image: highlightImg,
    products,
    presets: {},
    room_dimensions: roomDimensions,
    type: S.selectedSurface.type,
    openings,
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
  const MAX_WAIT = 10 * 60 * 1000; // 10 min
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
        showResult(data.result_images || { isometric: data.result_image });
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

const VIEW_LABELS = { isometric: 'Top View', front: 'Front View', corner: 'Corner View' };
const VIEW_ORDER  = ['isometric', 'front', 'corner'];

function buildCarousel(images, prefix) {
  const views = VIEW_ORDER.filter(k => images[k]);
  if (!views.length) return null;
  let idx = 0;

  const wrap = document.createElement('div');
  wrap.className = 'carousel-wrap';

  const label = document.createElement('div');
  label.className = 'carousel-label';

  const imgEl = document.createElement('img');
  imgEl.className = 'carousel-img';

  const dlLink = document.createElement('a');
  dlLink.className = 'carousel-dl';
  dlLink.textContent = '⬇ Download';

  const nav = document.createElement('div');
  nav.className = 'carousel-nav';

  const prevBtn = document.createElement('button');
  prevBtn.className = 'carousel-btn';
  prevBtn.textContent = '‹';

  const counter = document.createElement('span');
  counter.className = 'carousel-counter';

  const nextBtn = document.createElement('button');
  nextBtn.className = 'carousel-btn';
  nextBtn.textContent = '›';

  nav.append(prevBtn, counter, nextBtn);
  wrap.append(label, imgEl, dlLink, nav);

  function show(i) {
    idx = (i + views.length) % views.length;
    const key = views[idx];
    label.textContent = VIEW_LABELS[key] || key;
    imgEl.src = images[key];
    dlLink.href = images[key];
    dlLink.download = `${prefix}_${key}.png`;
    counter.textContent = `${idx + 1} / ${views.length}`;
    prevBtn.disabled = views.length <= 1;
    nextBtn.disabled = views.length <= 1;
  }

  prevBtn.addEventListener('click', () => show(idx - 1));
  nextBtn.addEventListener('click', () => show(idx + 1));
  show(0);
  return wrap;
}

function showResult(images) {
  resultSection.style.display = '';
  resultImg.style.display = 'none';
  resultDownload.style.display = 'none';
  const container = resultImg.parentElement;
  container.querySelectorAll('.carousel-wrap').forEach(el => el.remove());
  const carousel = buildCarousel(images, 'generated');
  if (carousel) container.appendChild(carousel);
}

// ── Mode tab switching ─────────────────────────────────────────
const tabSurface   = document.getElementById('tab-surface');
const tabFloorplan = document.getElementById('tab-floorplan');
const wsSurface    = document.getElementById('workspace-surface');
const wsFP         = document.getElementById('workspace-floorplan');

tabSurface.addEventListener('click', () => {
  tabSurface.classList.add('active');
  tabFloorplan.classList.remove('active');
  wsSurface.style.display = '';
  wsFP.style.display = 'none';
  resizeCanvases();
});

tabFloorplan.addEventListener('click', () => {
  tabFloorplan.classList.add('active');
  tabSurface.classList.remove('active');
  wsSurface.style.display = 'none';
  wsFP.style.display = '';
});

// ── Floor Plan JSON generation ────────────────────────────────
const DEMO_PAYLOAD = {
  "scene_id": "living_room_demo_001",
  "unit": "m",
  "room": {
    "width": 5,
    "length": 7,
    "height": 3,
    "flooring": {
      "type": "wood",
      "material": "light oak",
      "plank_direction": "horizontal",
      "plank_size": [1.2, 0.2]
    },
    "walls": {
      "thickness": 0.12,
      "color": "#F3EFE8"
    }
  },
  "openings": [
    {
      "id": "window_01",
      "type": "window",
      "wall": "north",
      "position_from_left": 1.7,
      "width": 1.6,
      "height": 0.9,
      "sill_height": 1.2,
      "frame_material": "oak wood"
    },
    {
      "id": "door_01",
      "type": "door",
      "wall": "south",
      "position_from_left": 2.05,
      "width": 0.9,
      "height": 2.2,
      "swing": "inward-left",
      "material": "oak wood"
    }
  ],
  "products": [
    {
      "id": "sofa_main",
      "category": "three_seater_sofa",
      "image_url": "https://www.customcreation.com.pk/wp-content/uploads/2025/01/SOFIA-%E2%80%93-Solid-Wood-Upholstered-3-Seater-Sofa-1.jpg",
      "dimensions": { "width": 2.2, "depth": 0.9, "height": 0.85 },
      "position": { "x": 0.45, "y": 3.5, "z": 0 },
      "rotation_y": 90,
      "material": { "fabric": "cream boucle" }
    },
    {
      "id": "bed_main",
      "category": "bed",
      "image_url": "https://furniturecity.com.pk/cdn/shop/files/image_1800x1800_a5d1c19e-294b-44a7-915f-67ba9ce9c301.jpg?v=1710531159",
      "dimensions": { "width": 1.6, "depth": 2.0, "height": 0.5 },
      "position": { "x": 2.5, "y": 1.0, "z": 0 },
      "rotation_y": 0,
      "material": { "fabric": "cream boucle" }
    }
  ],
  "lighting": {
    "type": "natural_daylight",
    "sun_direction": "north",
    "intensity": 0.85
  },
  "camera_views": [
    {
      "name": "front_view",
      "camera_position": [250, -220, 170],
      "target": [250, 540, 100],
      "fov": 35
    },
    {
      "name": "corner_view",
      "camera_position": [-180, -120, 190],
      "target": [250, 540, 100],
      "fov": 40
    }
  ]
};

const fpJsonEl       = document.getElementById('fp-json');
const fpSizeEl       = document.getElementById('fp-size');
const btnFpGenerate  = document.getElementById('btn-fp-generate');
const fpStatusTag    = document.getElementById('fp-status-tag');
const fpResultImg    = document.getElementById('fp-result-img');
const fpResultDl     = document.getElementById('fp-result-download');

function showFpResult(images) {
  fpResultImg.style.display = 'none';
  fpResultDl.style.display = 'none';
  const container = fpResultImg.parentElement;
  container.querySelectorAll('.carousel-wrap').forEach(el => el.remove());
  const carousel = buildCarousel(images, 'floor_plan');
  if (carousel) container.appendChild(carousel);
}
const fpPlaceholder  = document.getElementById('fp-placeholder');
const fpPromptText   = document.getElementById('fp-prompt-text');

fpJsonEl.value = JSON.stringify(DEMO_PAYLOAD, null, 2);

let fpGenerating = false;

btnFpGenerate.addEventListener('click', async () => {
  if (fpGenerating) return;

  let payload;
  try {
    payload = JSON.parse(fpJsonEl.value);
  } catch (e) {
    setFpStatus('Invalid JSON: ' + e.message, 'error');
    return;
  }

  payload.size = fpSizeEl.value;

  fpGenerating = true;
  btnFpGenerate.disabled = true;
  setFpStatus('Generating…', 'processing');
  fpPlaceholder.style.display = '';
  fpResultImg.classList.remove('visible');
  fpResultDl.style.display = 'none';
  fpPromptText.textContent = '';
  setStatus('Floor plan generation started…', 'processing');

  let genId;
  try {
    const res = await fetch('/api/floor-plan', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();
    genId = data.gen_id;
  } catch (err) {
    setFpStatus('Failed: ' + err.message, 'error');
    setStatus('Floor plan failed', 'error');
    fpGenerating = false;
    btnFpGenerate.disabled = false;
    return;
  }

  pollFpGeneration(genId);
});

async function pollFpGeneration(genId) {
  const INTERVAL = 2500;
  const MAX_WAIT = 10 * 60 * 1000;
  const start = Date.now();

  const poll = async () => {
    if (Date.now() - start > MAX_WAIT) {
      setFpStatus('Timed out', 'error');
      setStatus('Floor plan timed out', 'error');
      fpGenerating = false;
      btnFpGenerate.disabled = false;
      return;
    }
    try {
      const res = await fetch(`/api/generate/${genId}`);
      const data = await res.json();

      if (data.status === 'success') {
        setFpStatus('Done', 'success');
        setStatus('Floor plan complete!', 'success');
        fpPlaceholder.style.display = 'none';
        fpGenerating = false;
        btnFpGenerate.disabled = false;
        fetchFpPromptPreview(genId);
        showFpResult(data.result_images || { isometric: data.result_image });
      } else if (data.status === 'failed') {
        setFpStatus('Failed: ' + (data.reason || 'unknown'), 'error');
        setStatus('Floor plan failed', 'error');
        fpGenerating = false;
        btnFpGenerate.disabled = false;
      } else {
        setTimeout(poll, INTERVAL);
      }
    } catch {
      setTimeout(poll, INTERVAL);
    }
  };

  setTimeout(poll, INTERVAL);
}

async function fetchFpPromptPreview(genId) {
  // Show a truncated version of what was sent — parse the JSON and rebuild
  try {
    const payload = JSON.parse(fpJsonEl.value);
    // Ask backend for a preview via a lightweight endpoint if available;
    // for now just show a summary from the payload
    const room = payload.room || {};
    const n = (payload.products || []).length;
    fpPromptText.textContent =
      `Room ${room.width}×${room.length}${payload.unit}, ${room.height}${payload.unit} ceiling\n` +
      `${(payload.openings || []).length} opening(s), ${n} furniture piece(s)\n` +
      `Lighting: ${(payload.lighting || {}).type || 'N/A'}\n` +
      `View: ${((payload.camera_views || [])[0] || {}).name || 'N/A'}\n` +
      `Size: ${payload.size || '1024x1024'}`;
  } catch { /* ignore */ }
}

function setFpStatus(msg, cls = '') {
  fpStatusTag.textContent = msg;
  fpStatusTag.className = 'fp-status-tag' + (cls ? ' ' + cls : '');
}

// ── Sample test images ─────────────────────────────────────────
const SAMPLE_IMAGES = [
  {
    label: 'Sofia 3-Seater Sofa',
    url: 'https://www.customcreation.com.pk/wp-content/uploads/2025/01/SOFIA-%E2%80%93-Solid-Wood-Upholstered-3-Seater-Sofa-1.jpg',
  },
  {
    label: 'Furniture City Bed',
    url: 'https://furniturecity.com.pk/cdn/shop/files/image_1800x1800_a5d1c19e-294b-44a7-915f-67ba9ce9c301.jpg?v=1710531159',
  },
];

function renderSampleImages() {
  const container = document.getElementById('sample-images-list');
  if (!container) return;
  container.innerHTML = '';
  SAMPLE_IMAGES.forEach(si => {
    const div = document.createElement('div');
    div.className = 'sample-img-item';
    div.innerHTML = `
      <img src="${si.url}" alt="${si.label}" class="sample-img-thumb" />
      <div class="sample-img-info">
        <div class="sample-img-label">${si.label}</div>
        <button class="sample-img-use" data-url="${si.url}">Use →</button>
      </div>
    `;
    div.querySelector('.sample-img-use').addEventListener('click', evt => {
      const url = evt.target.dataset.url;
      // Assign to the first placed product without a URL
      const unset = S.placedProducts.find(pp => !pp.imageUrl.trim());
      if (unset) {
        unset.imageUrl = url;
        updatePlacedList();
        updateGenerateBtn();
        setStatus(`Assigned to ${unset.product.name}`, '');
      } else if (S.placedProducts.length > 0) {
        // Reassign the last product
        const last = S.placedProducts[S.placedProducts.length - 1];
        last.imageUrl = url;
        updatePlacedList();
        updateGenerateBtn();
        setStatus(`Assigned to ${last.product.name}`, '');
      } else {
        setStatus('Place a product first, then assign an image', '');
      }
    });
    container.appendChild(div);
  });
}

// ── Init ───────────────────────────────────────────────────────
loadCatalog();
resizeCanvases();
renderSampleImages();
