// DOM Elements
const fileInput = document.getElementById('fileInput');
const uploadZone = document.getElementById('uploadZone');
const uploadContent = document.getElementById('uploadContent');
const previewContainer = document.getElementById('previewContainer');
const imageCanvas = document.getElementById('imageCanvas');
const clearBtn = document.getElementById('clearBtn');
const detectBtn = document.getElementById('detectBtn');
const askBtn = document.getElementById('askBtn');
const loader = document.getElementById('loader');
const loaderText = document.getElementById('loaderText');
const apiStatus = document.getElementById('apiStatus');

let currentFile = null;
let currentImage = null;

// Initialization: check API health
async function checkHealth() {
    try {
        const res = await fetch('/health');
        if (res.ok) {
            apiStatus.textContent = 'API Online';
            apiStatus.classList.add('online');
        } else {
            apiStatus.textContent = 'API Degraded';
        }
    } catch (e) {
        apiStatus.textContent = 'API Offline';
    }
}
checkHealth();

// Tab Switching
document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
        document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
        
        btn.classList.add('active');
        document.getElementById(btn.dataset.target).classList.add('active');
    });
});

// Slider Update
const confThreshold = document.getElementById('confThreshold');
const confValue = document.getElementById('confValue');
confThreshold.addEventListener('input', (e) => {
    confValue.textContent = Number(e.target.value).toFixed(2);
});

// File Handling
function handleFile(file) {
    if (!file || !file.type.startsWith('image/')) return;
    
    currentFile = file;
    const reader = new FileReader();
    reader.onload = (e) => {
        const img = new Image();
        img.onload = () => {
            currentImage = img;
            drawOriginalImage();
            
            uploadContent.classList.add('hidden');
            previewContainer.classList.remove('hidden');
            detectBtn.disabled = false;
            askBtn.disabled = false;
        };
        img.src = e.target.result;
    };
    reader.readAsDataURL(file);
}

function drawOriginalImage() {
    if (!currentImage) return;
    const ctx = imageCanvas.getContext('2d');
    
    // Set canvas dimensions to match image natural size for crisp rendering
    imageCanvas.width = currentImage.naturalWidth;
    imageCanvas.height = currentImage.naturalHeight;
    
    ctx.drawImage(currentImage, 0, 0);
}

// Drag & Drop
uploadZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadZone.classList.add('dragover');
});
uploadZone.addEventListener('dragleave', () => {
    uploadZone.classList.remove('dragover');
});
uploadZone.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadZone.classList.remove('dragover');
    if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
});
fileInput.addEventListener('change', (e) => {
    if (e.target.files.length) handleFile(e.target.files[0]);
});

// Clear Image
clearBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    currentFile = null;
    currentImage = null;
    fileInput.value = '';
    
    uploadContent.classList.remove('hidden');
    previewContainer.classList.add('hidden');
    detectBtn.disabled = true;
    askBtn.disabled = true;
    
    document.getElementById('detectResults').innerHTML = '<div class="empty-state">Upload an image and run detection to see results here.</div>';
    document.getElementById('askResults').innerHTML = '<div class="empty-state">Upload an image and ask a question to see the AI reasoning here.</div>';
});

// Detection Logic
const classColors = {
    'Hardhat': '#3b82f6',
    'NO-Hardhat': '#ef4444',
    'Safety-Vest': '#10b981',
    'NO-Safety-Vest': '#f59e0b',
    'Person': '#8b5cf6'
};

detectBtn.addEventListener('click', async () => {
    if (!currentFile) return;
    
    loaderText.textContent = 'Running Object Detection...';
    loader.classList.remove('hidden');
    
    // Redraw image to clear previous boxes
    drawOriginalImage();
    
    const formData = new FormData();
    formData.append('file', currentFile);
    formData.append('conf_threshold', confThreshold.value);
    
    try {
        const res = await fetch('/detect', { method: 'POST', body: formData });
        const data = await res.json();
        
        if (res.ok) {
            drawBoxes(data.detections);
            renderDetectResults(data);
        } else {
            alert('Error: ' + data.detail);
        }
    } catch (err) {
        alert('Network Error: ' + err.message);
    } finally {
        loader.classList.add('hidden');
    }
});

function drawBoxes(detections) {
    if (!currentImage || !detections.length) return;
    const ctx = imageCanvas.getContext('2d');
    
    // Use line width proportional to image size
    const lineWidth = Math.max(3, Math.floor(currentImage.naturalWidth / 300));
    const fontSize = Math.max(14, Math.floor(currentImage.naturalWidth / 50));
    
    detections.forEach(det => {
        const [x1, y1, x2, y2] = det.bbox_xyxy;
        const width = x2 - x1;
        const height = y2 - y1;
        const color = classColors[det.class_name] || '#ffffff';
        
        // Draw Box
        ctx.strokeStyle = color;
        ctx.lineWidth = lineWidth;
        ctx.strokeRect(x1, y1, width, height);
        
        // Draw Label Background
        const text = `${det.class_name} ${(det.confidence * 100).toFixed(0)}%`;
        ctx.font = `600 ${fontSize}px Inter, sans-serif`;
        const textMetrics = ctx.measureText(text);
        
        ctx.fillStyle = color;
        ctx.fillRect(x1, y1 - fontSize - 8, textMetrics.width + 12, fontSize + 8);
        
        // Draw Text
        ctx.fillStyle = '#ffffff';
        ctx.fillText(text, x1 + 6, y1 - 6);
    });
}

function renderDetectResults(data) {
    const resDiv = document.getElementById('detectResults');
    const classCounts = {};
    data.detections.forEach(d => {
        classCounts[d.class_name] = (classCounts[d.class_name] || 0) + 1;
    });
    
    let html = `<div style="margin-bottom: 1rem;"><strong>Found ${data.detections.length} objects in ${data.inference_ms.toFixed(1)}ms</strong></div>`;
    
    if (data.detections.length === 0) {
        html += `<div class="empty-state">No objects detected above the threshold.</div>`;
    } else {
        Object.entries(classCounts).forEach(([label, count]) => {
            html += `<div class="metric-row">
                        <span style="display:flex; align-items:center; gap:8px;">
                            <span style="display:inline-block; width:12px; height:12px; border-radius:3px; background:${classColors[label]||'#fff'}"></span>
                            ${label}
                        </span>
                        <span class="metric-value">${count}</span>
                    </div>`;
        });
    }
    resDiv.innerHTML = html;
}

// Reasoning Logic
askBtn.addEventListener('click', async () => {
    if (!currentFile) return;
    const qInput = document.getElementById('questionInput').value.trim();
    if (!qInput) return alert('Please enter a question.');
    
    loaderText.textContent = 'Analyzing image & reasoning...';
    loader.classList.remove('hidden');
    
    const formData = new FormData();
    formData.append('file', currentFile);
    formData.append('question', qInput);
    
    try {
        const res = await fetch('/ask', { method: 'POST', body: formData });
        const data = await res.json();
        
        if (res.ok) {
            renderAskResults(data);
        } else {
            alert('Error: ' + data.detail);
        }
    } catch (err) {
        alert('Network Error: ' + err.message);
    } finally {
        loader.classList.add('hidden');
    }
});

function renderAskResults(data) {
    const resDiv = document.getElementById('askResults');
    
    const confClass = `confidence-${data.confidence}`;
    
    let html = `
        <div class="answer-box">
            <strong>AI Answer:</strong><br/>
            <span style="font-size: 1rem; line-height: 1.6; display: block; margin-top: 0.5rem;">${data.answer}</span>
        </div>
        
        <div class="metric-row">
            <span>Intent Routing</span>
            <span>${data.used_detector ? 'Used Object Detector' : 'Text Only (LLM)'}</span>
        </div>
        <div class="metric-row" style="margin-bottom: 1rem; border-bottom: none; font-size: 0.75rem; color: var(--text-secondary);">
            <em>Reason: ${data.route_reason}</em>
        </div>
        
        <div class="metric-row">
            <span>Guardrail Confidence</span>
            <span class="metric-value ${confClass}" style="text-transform: capitalize;">${data.confidence}</span>
        </div>
    `;
    
    if (data.evidence && data.confidence !== 'insufficient') {
        html += `
            <div style="margin-top: 1rem; padding-top: 1rem; border-top: 1px dashed var(--panel-border);">
                <strong>Detection Evidence Used:</strong>
                <div class="metric-row" style="margin-top:0.5rem"><span>Persons:</span><span>${data.evidence.persons_detected}</span></div>
                <div class="metric-row"><span>Hardhat Violations:</span><span>${data.evidence.hardhat_violations}</span></div>
                <div class="metric-row"><span>Vest Violations:</span><span>${data.evidence.vest_violations}</span></div>
            </div>
        `;
    }
    
    resDiv.innerHTML = html;
}
