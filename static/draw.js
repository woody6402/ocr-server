
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
let img = new Image();
let segments = [];

img.onload = () => {
  canvas.width = img.width;
  canvas.height = img.height;
  ctx.drawImage(img, 0, 0);
};

img.src = `/images/${imageName}`;

let startX, startY, isDrawing = false;

canvas.addEventListener('mousedown', e => {
  isDrawing = true;
  const rect = canvas.getBoundingClientRect();
  startX = e.clientX - rect.left;
  startY = e.clientY - rect.top;
});

canvas.addEventListener('mouseup', e => {
  if (!isDrawing) return;
  const rect = canvas.getBoundingClientRect();
  const endX = e.clientX - rect.left;
  const endY = e.clientY - rect.top;
  const x = Math.min(startX, endX);
  const y = Math.min(startY, endY);
  const w = Math.abs(startX - endX);
  const h = Math.abs(startY - endY);
  segments.push([x, y, w, h]);
  ctx.strokeStyle = "red";
  ctx.lineWidth = 2;
  ctx.strokeRect(x, y, w, h);
  isDrawing = false;
});

function saveSegments() {
  const identifier = document.getElementById("identifier").value;
  fetch("/save_segments", {
    method: "POST",
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ identifier, segments })
  }).then(res => res.json()).then(data => {
    alert("Gespeichert!");
  });
}
