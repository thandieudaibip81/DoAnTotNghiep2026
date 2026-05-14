/* ═══════════════════════════════════════════════════
   Fraud Guard Pro — Frontend Logic
   ═══════════════════════════════════════════════════ */

// ── DOM References ──
const predictionForm = document.getElementById('predictionForm');
const submitBtn = document.getElementById('submitBtn');
const resultCard = document.getElementById('resultCard');
const spinner = document.getElementById('spinner');
const verdictTitle = document.getElementById('verdictTitle');
const probabilityText = document.getElementById('probabilityText');
const probContainer = document.getElementById('probContainer');
const probBar = document.getElementById('probBar');
const errorMsg = document.getElementById('errorMsg');
const iconCircle = document.getElementById('resultIcon');
const timeInput = document.getElementById('Time');
const timeInMinutes = document.getElementById('timeInMinutes');

// ── State ──
let activeContext = null;
let currentSelectedId = null;
let currentFilter = 'all';
let selectedAiProvider = 'gemini';
let chatAiProvider = 'gemini';
window.allHistory = [];

// ── SVG Icons ──
const icons = {
    wait: '<svg fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>',
    fraud: '<svg fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"></path></svg>',
    safe: '<svg fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"></path></svg>',
    suspicious: '<svg fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8.228 9c.549-1.165 2.03-2 3.772-2 2.21 0 4 1.343 4 3 0 1.4-1.278 2.575-3.006 2.907-.542.104-.994.54-.994 1.093m0 3h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>',
};

// ══════════════════════════════════════════════════
// THEME
// ══════════════════════════════════════════════════

function toggleTheme() {
    const body = document.body;
    const newTheme = body.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
    body.setAttribute('data-theme', newTheme);
    localStorage.setItem('theme', newTheme);
    const icon = document.getElementById('themeIcon');
    icon.classList.toggle('fa-sun', newTheme === 'light');
    icon.classList.toggle('fa-moon', newTheme === 'dark');
}

(function initTheme() {
    const saved = localStorage.getItem('theme') || 'light';
    document.body.setAttribute('data-theme', saved);
    const icon = document.getElementById('themeIcon');
    if (icon) {
        icon.classList.toggle('fa-sun', saved === 'light');
        icon.classList.toggle('fa-moon', saved === 'dark');
    }
})();

// ══════════════════════════════════════════════════
// ABOUT DRAWER
// ══════════════════════════════════════════════════

function toggleAbout() {
    const drawer = document.getElementById('aboutDrawer');
    const overlay = document.getElementById('aboutOverlay');
    const isOpen = drawer.style.display === 'flex';
    drawer.style.display = isOpen ? 'none' : 'flex';
    overlay.style.display = isOpen ? 'none' : 'block';
    document.body.classList.toggle('about-active', !isOpen);
}

document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && document.body.classList.contains('about-active')) toggleAbout();
});

// ══════════════════════════════════════════════════
// MOBILE SIDEBAR
// ══════════════════════════════════════════════════

function toggleSidebar() {
    document.body.classList.toggle('mobile-sidebar-active');
}

// ══════════════════════════════════════════════════
// MODEL SELECTOR — load from backend
// ══════════════════════════════════════════════════

async function loadModels() {
    try {
        const res = await fetch('/models/');
        const data = await res.json();
        const select = document.getElementById('modelSelector');
        const badge = document.getElementById('modelBadge');

        select.innerHTML = data.models.map(m =>
            `<option value="${m.id}" ${m.id === data.default ? 'selected' : ''}>${m.display_name}</option>`
        ).join('');

        // Update badge
        const current = data.models.find(m => m.id === data.default);
        if (current) badge.textContent = current.sampling.toUpperCase();

        select.addEventListener('change', () => {
            const sel = data.models.find(m => m.id === select.value);
            if (sel) badge.textContent = sel.sampling.toUpperCase();
        });
    } catch (err) {
        console.error('Failed to load models:', err);
    }
}

// ══════════════════════════════════════════════════
// AI PROVIDER TOGGLE
// ══════════════════════════════════════════════════

function setAiProvider(provider, btn) {
    selectedAiProvider = provider;
    document.querySelectorAll('#aiProviderToggle .provider-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
}

function setChatProvider(provider, btn) {
    chatAiProvider = provider;
    document.querySelectorAll('#chatAiToggle .provider-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
}

// ══════════════════════════════════════════════════
// FORM AUTO-FILL
// ══════════════════════════════════════════════════

timeInput.addEventListener('input', (e) => {
    const seconds = parseFloat(e.target.value) || 0;
    timeInMinutes.innerText = `= (${(seconds / 60).toFixed(1)} phút)`;
});

function fillSafeData() {
    document.getElementById('Time').value = Math.floor(Math.random() * 50000);
    document.getElementById('Amount').value = (Math.random() * 80 + 10).toFixed(2);
    timeInput.dispatchEvent(new Event('input'));
    for (let i = 1; i <= 28; i++) {
        document.getElementById(`V${i}`).value = (Math.random() * 0.8 - 0.4).toFixed(4);
    }
}

function fillFraudData() {
    document.getElementById('Time').value = Math.floor(Math.random() * 172000);
    document.getElementById('Amount').value = (Math.random() * 1000 + 50).toFixed(2);
    timeInput.dispatchEvent(new Event('input'));

    const drivers = {
        V14: -8.5, V17: -12.0, V12: -7.2, V10: -5.5,
        V4: 4.8, V11: 4.1, V1: -4.5, V3: -5.0,
        V7: -3.5, V16: -4.0, V18: -2.8
    };

    for (let i = 1; i <= 28; i++) {
        const key = `V${i}`;
        const val = drivers[key]
            ? (drivers[key] + (Math.random() * 2 - 1)).toFixed(4)
            : (Math.random() * 3 - 1.5).toFixed(4);
        document.getElementById(key).value = val;
    }
}

// ══════════════════════════════════════════════════
// FORM VALIDATION
// ══════════════════════════════════════════════════

function validateInputs(data) {
    let ok = true;
    errorMsg.innerText = '';
    document.querySelectorAll('.input-group').forEach(el => el.classList.remove('invalid'));

    if (data.Time < 0 || data.Time > 172800) {
        const f = document.getElementById('Time');
        f.classList.add('invalid');
        f.closest('.input-group').classList.add('invalid');
        ok = false;
    }
    if (data.Amount < 0 || data.Amount > 25000) {
        const f = document.getElementById('Amount');
        f.classList.add('invalid');
        f.closest('.input-group').classList.add('invalid');
        ok = false;
    }
    for (let i = 1; i <= 28; i++) {
        if (data[`V${i}`] < -100 || data[`V${i}`] > 100) {
            const f = document.getElementById(`V${i}`);
            f.classList.add('invalid');
            f.closest('.input-group').classList.add('invalid');
            ok = false;
        }
    }
    if (!ok) errorMsg.innerText = 'Dữ liệu ngoài khoảng cho phép. Kiểm tra các ô đỏ.';
    return ok;
}

// ══════════════════════════════════════════════════
// PREDICTION
// ══════════════════════════════════════════════════

predictionForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const formData = new FormData(predictionForm);
    const txData = {};
    formData.forEach((v, k) => { txData[k] = parseFloat(v); });

    if (!validateInputs(txData)) return;

    const modelId = document.getElementById('modelSelector').value;

    // UI: loading state
    resultCard.classList.remove('state-safe', 'state-fraud', 'state-suspicious');
    document.getElementById('aiAnalysisBox').classList.add('hidden');
    document.getElementById('aiActionBar').classList.add('hidden');
    document.getElementById('aiActionBar').style.display = 'none';
    document.getElementById('modelUsedLabel').classList.add('hidden');
    
    document.getElementById('shapResultContainer').classList.add('hidden');
    document.getElementById('shapImage').classList.add('hidden');
    document.getElementById('shapImage').src = '';

    submitBtn.disabled = true;
    submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Đang phân tích...';
    iconCircle.innerHTML = '';
    spinner.style.display = 'block';
    verdictTitle.innerText = 'Đang xử lý...';
    probabilityText.innerText = 'Đang phân tích các đặc trưng PCA...';
    probContainer.classList.add('hidden');
    probBar.style.width = '0%';
    errorMsg.innerText = '';

    try {
        const res = await fetch('/predict/', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ transaction: txData, model_id: modelId }),
        });

        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.detail || 'Không thể phân tích giao dịch.');
        }

        const data = await res.json();
        activeContext = { ...txData, ...data };

        setTimeout(() => {
            renderResult(data);
            loadHistory();
        }, 600);
    } catch (err) {
        spinner.style.display = 'none';
        iconCircle.innerHTML = icons.wait;
        verdictTitle.innerText = 'Phân tích Thất bại';
        probabilityText.innerText = 'Kiểm tra lại kết nối hoặc log server.';
        errorMsg.innerText = err.message;
        submitBtn.disabled = false;
        submitBtn.innerHTML = '<i class="fas fa-search"></i> Bắt đầu Dự đoán';
    }
});

function renderResult(data) {
    spinner.style.display = 'none';
    submitBtn.disabled = false;
    submitBtn.innerHTML = '<i class="fas fa-search"></i> Bắt đầu Dự đoán';

    probContainer.classList.remove('hidden');
    const prob = data.probability;
    const pct = (prob * 100).toFixed(2);

    resultCard.classList.remove('state-safe', 'state-fraud', 'state-suspicious');

    if (prob >= 0.75) {
        probBar.style.width = `${pct}%`;
        resultCard.classList.add('state-fraud');
        iconCircle.innerHTML = icons.fraud;
        verdictTitle.innerText = 'Phát hiện Gian lận!';
        probabilityText.innerHTML = `Mức độ Rủi ro: <strong>${pct}%</strong><br><small>Giao dịch có dấu hiệu bất thường nghiêm trọng.</small>`;
    } else if (prob >= 0.35) {
        probBar.style.width = `${pct}%`;
        resultCard.classList.add('state-suspicious');
        iconCircle.innerHTML = icons.suspicious;
        verdictTitle.innerText = 'Giao dịch Nghi ngờ';
        probabilityText.innerHTML = `Xác suất Nghi ngờ: <strong>${pct}%</strong><br><small>Phát hiện một số yếu tố không chắc chắn.</small>`;
    } else {
        const safe = ((1 - prob) * 100).toFixed(2);
        probBar.style.width = `${safe}%`;
        resultCard.classList.add('state-safe');
        iconCircle.innerHTML = icons.safe;
        verdictTitle.innerText = 'Giao dịch An toàn';
        probabilityText.innerHTML = `Độ tin cậy: <strong>${safe}%</strong><br><small>Không tìm thấy dấu hiệu gian lận.</small>`;
    }

    // Show model used
    if (data.model_used) {
        const label = document.getElementById('modelUsedLabel');
        label.textContent = `Mô hình: ${data.model_used}`;
        label.classList.remove('hidden');
    }

    // Show AI action bar
    const aiBar = document.getElementById('aiActionBar');
    aiBar.classList.remove('hidden');
    aiBar.style.display = 'flex';
}

// ══════════════════════════════════════════════════
// AI ANALYSIS
// ══════════════════════════════════════════════════

function triggerAIAnalysis() {
    if (!activeContext) return;
    const btn = document.getElementById('aiAnalyzeBtn');
    btn.disabled = true;
    btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Đang phân tích...';
    fetchAIAnalysis(activeContext);
}

async function fetchAIAnalysis(data) {
    const box = document.getElementById('aiAnalysisBox');
    const txt = document.getElementById('aiAnalysisText');
    const btn = document.getElementById('aiAnalyzeBtn');

    box.classList.remove('hidden');
    txt.innerHTML = '<i class="fas fa-spinner fa-spin"></i> AI đang phân tích...';

    try {
        const res = await fetch('/analyze/', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                data: activeContext,
                prediction: parseInt(data.prediction || data.fraud_prediction || 0),
                probability: parseFloat(data.probability || 0),
                model_used: data.model_used || '',
                provider: selectedAiProvider,
            }),
        });
        const result = await res.json();
        txt.innerText = result.analysis;
    } catch (e) {
        txt.innerText = 'Không thể kết nối AI. Thử chuyển sang provider khác.';
    } finally {
        btn.disabled = false;
        btn.innerHTML = '<i class="fas fa-magic"></i> Phân tích bằng AI';
    }
}

async function triggerSHAPAnalysis() {
    if (!activeContext) return;
    const btn = document.getElementById('shapAnalyzeBtn');
    const container = document.getElementById('shapResultContainer');
    const spinner = document.getElementById('shapSpinner');
    const img = document.getElementById('shapImage');
    
    btn.disabled = true;
    btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Đang vẽ biểu đồ...';
    container.classList.remove('hidden');
    img.classList.add('hidden');
    spinner.style.display = 'block';

    try {
        const modelId = document.getElementById('modelSelector').value;
        const txData = {};
        for (let k in activeContext) {
            if (["Time", "Amount"].includes(k) || k.startsWith("V")) {
                txData[k] = activeContext[k];
            }
        }

        const res = await fetch('/explain/', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ transaction: txData, model_id: modelId }),
        });

        if (!res.ok) throw new Error('Không thể tạo biểu đồ SHAP');
        const data = await res.json();
        
        if (data.success) {
            img.src = data.image_base64;
            img.classList.remove('hidden');
        } else {
            alert('Lỗi: ' + data.error);
        }
    } catch (e) {
        alert('Lỗi kết nối khi gọi /explain/');
    } finally {
        spinner.style.display = 'none';
        btn.disabled = false;
        btn.innerHTML = '<i class="fas fa-chart-bar"></i> Xem Biểu Đồ Giải Thích Toán Học';
    }
}

// ══════════════════════════════════════════════════
// HISTORY
// ══════════════════════════════════════════════════

function setHistoryFilter(type, btn) {
    currentFilter = type;
    document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    renderHistory(window.allHistory);
}

async function loadHistory() {
    try {
        const res = await fetch('/history/');
        window.allHistory = await res.json();
        renderHistory(window.allHistory);
    } catch (err) {
        console.error('Failed to load history:', err);
    }
}

function renderHistory(history) {
    const list = document.getElementById('historyList');
    if (!list) return;

    const filtered = history.filter(item => {
        if (currentFilter === 'all') return true;
        const isFraud = item.verdict === 'Gian lận' || item.prediction === 1;
        if (currentFilter === 'fraud') return isFraud;
        return item.verdict === 'An toàn' || item.prediction === 0;
    });

    if (filtered.length === 0) {
        list.innerHTML = '<div class="history-empty">Không tìm thấy dữ liệu.</div>';
        return;
    }

    list.innerHTML = filtered.map((item, index) => {
        let cls = 'status-safe';
        if (item.verdict === 'Gian lận') cls = 'status-fraud';
        else if (item.verdict === 'Nghi ngờ') cls = 'status-suspicious';

        const displayIndex = filtered.length - index;

        return `
            <div class="history-item ${item.id === currentSelectedId ? 'active' : ''}" data-id="${item.id}" onclick="selectHistoryItem(${item.id})">
                <div class="item-header">
                    <span>#${displayIndex}</span>
                    <span style="flex:1; text-align:right; margin-right:8px;">${item.timestamp}</span>
                    <button class="delete-btn" onclick="deleteHistoryItem(event, ${item.id})" title="Xóa giao dịch">
                        <i class="fas fa-times"></i>
                    </button>
                </div>
                <div class="item-body">
                    <div>
                        <span class="item-amount">$${item.amount.toLocaleString()}</span>
                        <div class="item-model">${item.model_used || ''}</div>
                    </div>
                    <div class="status-dot ${cls}"></div>
                </div>
            </div>
        `;
    }).join('');

    // Update chat context selector
    const sel = document.getElementById('chatContextSelector');
    if (sel) {
        const cur = sel.value;
        sel.innerHTML = '<option value="">-- Chọn giao dịch --</option>' +
            filtered.map((item, index) => `<option value="${item.id}">GD #${filtered.length - index} - $${item.amount.toLocaleString()}</option>`).join('');
        sel.value = cur;
    }
}

async function deleteHistoryItem(event, id) {
    event.stopPropagation(); // Ngăn sự kiện click lan ra lịch sử giao dịch
    if (!confirm('Bạn có chắc chắn muốn xóa giao dịch này khỏi hệ thống?')) return;
    
    try {
        const res = await fetch(`/history/${id}`, { method: 'DELETE' });
        if (res.ok) {
            window.allHistory = window.allHistory.filter(h => h.id !== id);
            if (currentSelectedId === id) resetUI();
            renderHistory(window.allHistory);
        } else {
            alert('Lỗi: Không thể xóa giao dịch này.');
        }
    } catch (e) {
        console.error('Delete error:', e);
        alert('Lỗi kết nối khi xóa giao dịch.');
    }
}

async function selectHistoryItem(id) {
    if (currentSelectedId === id) { resetUI(); return; }
    currentSelectedId = id;
    document.querySelectorAll('.history-item').forEach(el => el.classList.remove('active'));
    const target = document.querySelector(`.history-item[data-id="${id}"]`);
    if (target) target.classList.add('active');

    // Auto close sidebar on mobile after selection
    document.body.classList.remove('mobile-sidebar-active');

    try {
        const res = await fetch(`/history/${id}`);
        const data = await res.json();

        document.getElementById('Time').value = data.Time;
        document.getElementById('Amount').value = data.Amount;
        for (let i = 1; i <= 28; i++) {
            document.getElementById(`V${i}`).value = data.V[`V${i}`];
        }
        timeInput.dispatchEvent(new Event('input'));

        activeContext = data;
        renderResult({
            verdict: data.verdict,
            probability: data.probability,
            prediction: data.prediction,
            model_used: data.model_used,
        });
    } catch (e) {
        console.error('Select history item failed', e);
    }
}

function resetUI() {
    currentSelectedId = null;
    document.querySelectorAll('.history-item').forEach(el => el.classList.remove('active'));
    predictionForm.reset();
    timeInput.dispatchEvent(new Event('input'));

    verdictTitle.innerText = 'Đang chờ Dữ liệu';
    probabilityText.innerText = 'Nhập thông số để chạy mô hình ML...';
    resultCard.classList.remove('state-safe', 'state-fraud', 'state-suspicious');
    probContainer.classList.add('hidden');
    iconCircle.style.display = 'flex';
    iconCircle.innerHTML = icons.wait;

    document.getElementById('aiAnalysisBox').classList.add('hidden');
    document.getElementById('modelUsedLabel').classList.add('hidden');

    const aiBar = document.getElementById('aiActionBar');
    aiBar.classList.add('hidden');
    aiBar.style.display = 'none';

    const btn = document.getElementById('aiAnalyzeBtn');
    btn.disabled = false;
    btn.innerHTML = '<i class="fas fa-robot"></i> Phân tích bằng AI';

    activeContext = null;
}

// ══════════════════════════════════════════════════
// CHATBOT
// ══════════════════════════════════════════════════

function toggleChat() {
    const win = document.getElementById('chatWindow');
    win.classList.toggle('hidden');
    if (!win.classList.contains('hidden')) loadHistory();
}

function resetChat() {
    const body = document.getElementById('chatBody');
    body.innerHTML = '<div class="message ai-msg">Hội thoại đã được làm mới. Tôi có thể giúp gì thêm?</div>';
    activeContext = null;
    document.getElementById('chatContextSelector').value = '';
}

async function updateChatContext() {
    const id = document.getElementById('chatContextSelector').value;
    if (!id) { activeContext = null; return; }

    try {
        const res = await fetch(`/history/${id}`);
        const data = await res.json();
        activeContext = data;

        const body = document.getElementById('chatBody');
        const div = document.createElement('div');
        div.className = 'message ai-msg';
        div.style.background = 'rgba(56, 189, 248, 0.1)';
        div.style.border = '1px dashed var(--primary-btn)';
        div.innerHTML = `<i>Đã chọn giao dịch #${id} ($${data.Amount}) làm ngữ cảnh.</i>`;
        body.appendChild(div);
        body.scrollTop = body.scrollHeight;
    } catch (e) {
        console.error('Update context failed', e);
    }
}

async function sendChatMessage() {
    const input = document.getElementById('chatInput');
    const body = document.getElementById('chatBody');
    const msg = input.value.trim();
    if (!msg) return;

    // User message
    const userDiv = document.createElement('div');
    userDiv.className = 'message user-msg';
    userDiv.innerText = msg;
    body.appendChild(userDiv);
    input.value = '';
    body.scrollTop = body.scrollHeight;

    // Typing indicator
    const typingDiv = document.createElement('div');
    typingDiv.className = 'message ai-msg';
    typingDiv.innerHTML = '<i class="fas fa-circle-notch fa-spin"></i> ...';
    body.appendChild(typingDiv);
    body.scrollTop = body.scrollHeight;

    try {
        const res = await fetch('/chat/', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                message: msg,
                provider: chatAiProvider,
                context: activeContext,
            }),
        });
        const data = await res.json();
        typingDiv.innerText = data.response;
    } catch (e) {
        typingDiv.innerText = 'Xin lỗi, đang gặp trục trặc kỹ thuật. Thử chuyển AI provider.';
    }
    body.scrollTop = body.scrollHeight;
}

document.getElementById('chatInput').addEventListener('keypress', (e) => {
    if (e.key === 'Enter') sendChatMessage();
});

// ══════════════════════════════════════════════════
// MODE SWITCHING (Demo / Nghiệp vụ)
// ══════════════════════════════════════════════════

let currentMode = 'demo';

function setMode(mode) {
    currentMode = mode;
    document.getElementById('demoMode').classList.toggle('hidden', mode !== 'demo');
    document.getElementById('nghiepvuMode').classList.toggle('hidden', mode !== 'nghiepvu');
    document.getElementById('modeBtnDemo').classList.toggle('active', mode === 'demo');
    document.getElementById('modeBtnNghiepvu').classList.toggle('active', mode === 'nghiepvu');
    
    // Manage Chatbots visibility based on mode
    document.getElementById('chatbotSystem').classList.toggle('hidden', mode !== 'demo');
    document.getElementById('chatbotSystemNV').classList.toggle('hidden', mode !== 'nghiepvu');

    // Show/hide sidebar based on mode
    const sidebar = document.getElementById('historySidebar');
    if (sidebar) sidebar.classList.toggle('hidden', mode === 'nghiepvu');
    if (mode === 'nghiepvu') {
        loadSampleFiles();
        loadBatchHistory();
    }
}

// ══════════════════════════════════════════════════
// NGHIỆP VỤ — TAB SWITCHING
// ══════════════════════════════════════════════════

function setNVTab(tab, btn) {
    document.getElementById('nvImportTab').classList.toggle('hidden', tab !== 'import');
    document.getElementById('nvManageTab').classList.toggle('hidden', tab !== 'manage');
    document.querySelectorAll('.nv-tab').forEach(b => b.classList.remove('active'));
    if (btn) btn.classList.add('active');
    if (tab === 'manage') loadBatchHistory();
}

// ══════════════════════════════════════════════════
// FILE UPLOAD (Drag & Drop)
// ══════════════════════════════════════════════════

let selectedFile = null;

function initDropZone() {
    const dz = document.getElementById('dropZone');
    const fi = document.getElementById('fileInput');
    if (!dz || !fi) return;

    dz.addEventListener('click', () => fi.click());
    dz.addEventListener('dragover', e => { e.preventDefault(); dz.classList.add('drag-over'); });
    dz.addEventListener('dragleave', () => dz.classList.remove('drag-over'));
    dz.addEventListener('drop', e => {
        e.preventDefault(); dz.classList.remove('drag-over');
        if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
    });
    fi.addEventListener('change', () => { if (fi.files.length) handleFile(fi.files[0]); });
}

function handleFile(file) {
    const ext = file.name.split('.').pop().toLowerCase();
    const errEl = document.getElementById('batchError');
    if (!['xlsx', 'csv'].includes(ext)) {
        errEl.textContent = `Lỗi: Định dạng ".${ext}" không hỗ trợ. Chỉ chấp nhận .xlsx hoặc .csv`;
        return;
    }
    errEl.textContent = '';
    selectedFile = file;
    document.getElementById('fileInfo').classList.remove('hidden');
    document.getElementById('fileName').textContent = `${file.name} (${(file.size/1024).toFixed(1)} KB)`;
    document.getElementById('batchPredictBtn').disabled = false;
}

function clearFile() {
    selectedFile = null;
    document.getElementById('fileInput').value = '';
    document.getElementById('fileInfo').classList.add('hidden');
    document.getElementById('batchPredictBtn').disabled = true;
    document.getElementById('batchError').textContent = '';
}

// ══════════════════════════════════════════════════
// BATCH PREDICTION
// ══════════════════════════════════════════════════

async function runBatchPredict() {
    if (!selectedFile) return;
    const btn = document.getElementById('batchPredictBtn');
    const errEl = document.getElementById('batchError');
    btn.disabled = true;
    btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Đang phân tích...';
    errEl.textContent = '';

    const modelId = document.getElementById('batchModelSelector').value;
    const formData = new FormData();
    formData.append('file', selectedFile);
    formData.append('model_id', modelId);

    try {
        const res = await fetch('/batch-predict/', { method: 'POST', body: formData });
        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.detail || 'Lỗi không xác định');
        }
        const data = await res.json();
        currentBatchContextId = data.batch_id;
        const selNV = document.getElementById('chatContextSelectorNV');
        if (selNV) selNV.value = data.batch_id;
        renderBatchResults(data);
    } catch (e) {
        errEl.textContent = `❌ ${e.message}`;
    } finally {
        btn.disabled = false;
        btn.innerHTML = '<i class="fas fa-play"></i> Chạy Dự đoán';
    }
}

let lastBatchResults = null;  // Store batch results globally for SHAP
let lastBatchModelId = 'random_forest_smote';

function renderBatchResults(data) {
    document.getElementById('batchResultsPlaceholder').classList.add('hidden');
    document.getElementById('batchResults').classList.remove('hidden');
    lastBatchResults = data;
    lastBatchModelId = document.getElementById('batchModelSelector').value || 'random_forest_smote';

    const hasFraud = data.fraud > 0;
    const summaryClass = hasFraud ? 'summary-fraud' : 'summary-safe';
    const conclusion = hasFraud
        ? `⚠️ Phát hiện ${data.fraud} giao dịch bất thường — Cần kiểm tra`
        : '✅ Tất cả giao dịch đều an toàn';

    document.getElementById('batchSummary').innerHTML = `
        <div class="summary-header ${summaryClass}">
            <div class="summary-customer">
                <i class="fas fa-user"></i> <strong>${data.customer_name || 'N/A'}</strong>
                <span class="summary-bank">${data.bank_name || ''}</span>
            </div>
            <div class="summary-model">Mô hình: ${data.model_used}</div>
        </div>
        <div class="summary-stats">
            <div class="stat-box"><div class="stat-num">${data.total}</div><div class="stat-label">Tổng GD</div></div>
            <div class="stat-box stat-safe"><div class="stat-num">${data.safe}</div><div class="stat-label">🟢 An toàn</div></div>
            <div class="stat-box stat-suspect"><div class="stat-num">${data.suspect}</div><div class="stat-label">⚠️ Nghi ngờ</div></div>
            <div class="stat-box stat-fraud"><div class="stat-num">${data.fraud}</div><div class="stat-label">🔴 Gian lận</div></div>
        </div>
        <div class="summary-conclusion ${summaryClass}">${conclusion}</div>`;

    const tbody = document.getElementById('batchTableBody');
    tbody.innerHTML = data.results.map((r, idx) => {
        const cls = r.verdict === 'Gian lận' ? 'row-fraud' : (r.verdict === 'Nghi ngờ' ? 'row-suspect' : '');
        const icon = r.verdict === 'Gian lận' ? '🔴' : (r.verdict === 'Nghi ngờ' ? '⚠️' : '🟢');
        const confLabel = `${r.verdict} ${(r.probability*100).toFixed(1)}%`;
        const amtDisplay = r.amount_display ? Number(r.amount_display).toLocaleString('vi-VN') : r.amount;
        return `<tr class="${cls}">
            <td>${r.row}</td><td>${r.date}</td><td>${r.time || ''}</td><td>${r.description}</td>
            <td>${amtDisplay}</td>
            <td><strong>${icon} ${r.verdict}</strong></td>
            <td>${confLabel}</td>
            <td><button class="shap-explain-btn" onclick="openShapModal('batch', ${idx})">🔍 Giải thích</button></td>
        </tr>`;
    }).join('');
}

// ══════════════════════════════════════════════════
// SAMPLE FILES
// ══════════════════════════════════════════════════

async function loadSampleFiles() {
    try {
        const res = await fetch('/sample-files/');
        const data = await res.json();
        const list = document.getElementById('sampleFilesList');
        if (!list) return;
        list.innerHTML = data.files.map(f =>
            `<a href="/sample-files/${f.name}" class="sample-file-link" download>
                <i class="fas fa-file-excel"></i> ${f.name}
                <small>(${(f.size/1024).toFixed(0)} KB)</small>
            </a>`
        ).join('');
    } catch (e) { console.error('Sample files error:', e); }
}

// ══════════════════════════════════════════════════
// BATCH HISTORY (Quản lý Giao dịch)
// ══════════════════════════════════════════════════

async function loadBatchHistory() {
    try {
        const res = await fetch('/batch-history/');
        const data = await res.json();
        const list = document.getElementById('batchHistoryList');
        if (!list) return;
        if (data.length === 0) {
            list.innerHTML = '<div class="history-empty">Chưa có dữ liệu import...</div>';
            return;
        }
        list.innerHTML = data.map((b, index) => {
            const hasFraud = b.fraud > 0;
            const batchNum = data.length - index;
            return `<div class="batch-item ${hasFraud ? 'batch-has-fraud' : ''}" onclick="viewBatchDetail('${b.batch_id}')">
                <div class="batch-item-header">
                    <div>
                        <span style="display:inline-block; background:#e2e8f0; color:#475569; padding:2px 8px; border-radius:12px; font-size:0.8rem; font-weight:bold; margin-right:8px;">Lô #${batchNum}</span>
                        <strong><i class="fas fa-user"></i> ${b.customer_name || 'N/A'}</strong>
                        <span style="margin-left:8px; color:#64748b;">${b.bank_name || ''}</span>
                    </div>
                    <div style="display:flex; align-items:center; gap:15px;">
                        <span style="font-size: 0.85rem; color: #3b82f6; font-weight: 500;">Mô hình: ${b.model_used || 'N/A'}</span>
                        <button class="delete-btn" onclick="event.stopPropagation();deleteBatch('${b.batch_id}')" title="Xóa">
                            <i class="fas fa-trash"></i></button>
                    </div>
                </div>
                <div class="batch-item-stats">
                    <span>📊 ${b.total} GD</span>
                    <span>🟢 ${b.safe}</span>
                    <span>🔴 ${b.fraud}</span>
                    <span class="batch-date">${b.imported_at}</span>
                </div>
            </div>`;
        }).join('');
        
        // Populate chat NV context selector
        const selNV = document.getElementById('chatContextSelectorNV');
        if (selNV) {
            selNV.innerHTML = '<option value="">-- Chọn lô giao dịch --</option>' +
                data.map((b, index) => `<option value="${b.batch_id}">Lô #${data.length - index} - ${b.customer_name} (${b.bank_name})</option>`).join('');
            if (currentBatchContextId) selNV.value = currentBatchContextId;
        }
    } catch (e) { console.error('Batch history error:', e); }
}

async function viewBatchDetail(batchId) {
    try {
        const res = await fetch(`/batch-history/${batchId}`);
        const data = await res.json();
        const panel = document.getElementById('batchDetailPanel');
        panel.classList.remove('hidden');
        currentBatchContextId = batchId;
        const selNV = document.getElementById('chatContextSelectorNV');
        if (selNV) selNV.value = batchId;
        const first = data[0] || {};
        const fraud = data.filter(r => r.verdict === 'Gian lận').length;
        document.getElementById('batchDetailSummary').innerHTML = `
            <div class="summary-header ${fraud > 0 ? 'summary-fraud' : 'summary-safe'}">
                <strong>${first.customer_name || 'N/A'}</strong> — ${first.bank_name || ''}
                <span style="margin-left:auto;">${data.length} giao dịch | 🔴 ${fraud} gian lận</span>
            </div>`;
        document.getElementById('batchDetailBody').innerHTML = data.map((r, i) => {
            const cls = r.verdict === 'Gian lận' ? 'row-fraud' : (r.verdict === 'Nghi ngờ' ? 'row-suspect' : '');
            const icon = r.verdict === 'Gian lận' ? '🔴' : (r.verdict === 'Nghi ngờ' ? '⚠️' : '🟢');
            return `<tr class="${cls}"><td>${i+1}</td><td>${r.description}</td><td>$${r.amount}</td>
                <td>${icon} ${r.verdict}</td><td>${(r.probability*100).toFixed(1)}%</td>
                <td><button class="shap-explain-btn" onclick="openShapModal('detail', ${i}, '${batchId}')">🔍 Giải thích</button></td></tr>`;
        }).join('');
        
        // Store detail data globally
        window._batchDetailData = data;
        window._batchDetailBatchId = batchId;
        
        panel.scrollIntoView({ behavior: 'smooth' });
    } catch (e) { console.error('Batch detail error:', e); }
}

async function deleteBatch(batchId) {
    if (!confirm('Xóa tất cả giao dịch trong batch này?')) return;
    try {
        await fetch(`/batch-history/${batchId}`, { method: 'DELETE' });
        loadBatchHistory();
        document.getElementById('batchDetailPanel').classList.add('hidden');
    } catch (e) { console.error('Delete batch error:', e); }
}

// ══════════════════════════════════════════════════
// INIT
// ══════════════════════════════════════════════════

document.addEventListener('DOMContentLoaded', () => {
    loadModels();
    loadHistory();
    initDropZone();
    // Populate batch model selector too
    const bms = document.getElementById('batchModelSelector');
    if (bms) {
        fetch('/models/').then(r => r.json()).then(data => {
            bms.innerHTML = data.models.map(m =>
                `<option value="${m.id}" ${m.id === data.default ? 'selected' : ''}>${m.display_name}</option>`
            ).join('');
        }).catch(() => {});
    }
});

// ══════════════════════════════════════════════════
// AI CHATBOT & ANALYSIS (NGHIỆP VỤ MODE)
// ══════════════════════════════════════════════════

let currentBatchContextId = null;
let currentChatProviderNV = 'gemini';
let currentAiProviderNV = 'gemini';

function toggleChatNV() {
    const w = document.getElementById('chatWindowNV');
    w.classList.toggle('hidden');
    if (!w.classList.contains('hidden') && document.getElementById('chatBodyNV').children.length <= 1) {
        addChatMessageNV('Xin chào! Tôi có thể phân tích lô giao dịch hiện tại giúp bạn. Bạn cần hỏi gì?', false);
    }
}

function resetChatNV() {
    document.getElementById('chatBodyNV').innerHTML = '';
    addChatMessageNV('Lịch sử trò chuyện đã được làm mới. Bạn cần hỗ trợ gì về các lô giao dịch?', false);
}

function setChatProviderNV(provider, btn) {
    currentChatProviderNV = provider;
    document.querySelectorAll('#chatAiToggleNV .provider-btn').forEach(b => {
        if (b.dataset.provider === provider) {
            b.classList.add('active');
        } else {
            b.classList.remove('active');
        }
    });
}

function setAiProviderNV(provider, btn) {
    currentAiProviderNV = provider;
    document.querySelectorAll('#chatAiToggleNV_import .provider-btn, #chatAiToggleNV_manage .provider-btn').forEach(b => {
        if (b.dataset.provider === provider) {
            b.classList.add('active');
        } else {
            b.classList.remove('active');
        }
    });
}

function addChatMessageNV(msg, isUser=false) {
    const body = document.getElementById('chatBodyNV');
    const div = document.createElement('div');
    div.className = `message ${isUser ? 'user-msg' : 'ai-msg'}`;
    if (isUser) {
        div.textContent = msg;
    } else {
        div.innerHTML = marked.parse(msg);
    }
    body.appendChild(div);
    body.scrollTop = body.scrollHeight;
}

async function sendChatMessageNV() {
    const input = document.getElementById('chatInputNV');
    const msg = input.value.trim();
    if (!msg) return;

    addChatMessageNV(msg, true);
    input.value = '';

    const payload = {
        message: msg,
        provider: currentChatProviderNV,
        batch_id: currentBatchContextId || ""
    };

    // Show typing indicator
    const typingDiv = document.createElement('div');
    typingDiv.className = 'message ai-msg typing-indicator';
    typingDiv.innerHTML = '<i class="fas fa-ellipsis-h fa-fade"></i>';
    document.getElementById('chatBodyNV').appendChild(typingDiv);

    try {
        const res = await fetch('/chat-batch/', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        document.getElementById('chatBodyNV').removeChild(typingDiv);
        const data = await res.json();
        addChatMessageNV(data.response, false);
    } catch (e) {
        if (document.getElementById('chatBodyNV').contains(typingDiv)) {
            document.getElementById('chatBodyNV').removeChild(typingDiv);
        }
        addChatMessageNV('Xin lỗi, đã có lỗi kết nối đến AI. Vui lòng thử lại.', false);
        console.error(e);
    }
}

document.getElementById('chatInputNV')?.addEventListener('keypress', function (e) {
    if (e.key === 'Enter') sendChatMessageNV();
});

function updateChatContextNV() {
    const sel = document.getElementById('chatContextSelectorNV');
    currentBatchContextId = sel.value || null;
    resetChatNV();
}

async function triggerBatchAIAnalysis() {
    if (!currentBatchContextId) {
        alert("Vui lòng tải lên một lô giao dịch hoặc chọn một lô giao dịch từ phần Quản lý trước.");
        return;
    }
    
    let container = document.getElementById('aiExplanationNV_import');
    let content = document.getElementById('aiExplanationContentNV_import');
    if (!document.getElementById('nvManageTab').classList.contains('hidden')) {
        container = document.getElementById('aiExplanationNV_manage');
        content = document.getElementById('aiExplanationContentNV_manage');
    }
    
    container.classList.remove('hidden');
    content.innerHTML = '<div class="spinner" style="margin: 20px auto;"></div><p style="text-align:center;">AI đang phân tích tổng quan...</p>';

    try {
        const res = await fetch('/analyze-batch/', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ batch_id: currentBatchContextId, provider: currentAiProviderNV })
        });
        if (!res.ok) throw new Error("Analysis failed");
        const data = await res.json();
        content.innerText = data.analysis;
    } catch (e) {
        content.innerHTML = `<p style="color:#ef4444; text-align:center;">Lỗi hệ thống: ${e.message}. Vui lòng thử API khác.</p>`;
        console.error(e);
    }
}

// ══════════════════════════════════════════════════
// SHAP EXPLANATION MODAL
// ══════════════════════════════════════════════════

let shapModalProvider = 'gemini';
let shapModalTxData = null;  // Store current tx data for re-analysis

function setShapModalProvider(provider, btn) {
    shapModalProvider = provider;
    document.querySelectorAll('#shapModalAiToggle .provider-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    // Re-run AI analysis if we have data
    if (shapModalTxData) {
        runShapAIAnalysis(shapModalTxData);
    }
}

function closeShapModal() {
    document.getElementById('shapModal').classList.add('hidden');
    shapModalTxData = null;
}

async function openShapModal(source, index, batchId) {
    const modal = document.getElementById('shapModal');
    const spinnerEl = document.getElementById('shapModalSpinner');
    const imgEl = document.getElementById('shapModalImage');
    const aiText = document.getElementById('shapModalAiText');

    // Reset state
    modal.classList.remove('hidden');
    spinnerEl.style.display = 'flex';
    imgEl.classList.add('hidden');
    imgEl.src = '';
    aiText.innerHTML = '<p style="color: var(--text-muted);">Đang chờ biểu đồ SHAP...</p>';

    // Get transaction data
    let txRow = null;
    let modelId = 'random_forest_smote'; // default fallback

    if (source === 'batch' && lastBatchResults) {
        txRow = lastBatchResults.results[index];
        modelId = lastBatchModelId; // This is set in renderBatchResults
    } else if (source === 'detail' && window._batchDetailData) {
        txRow = window._batchDetailData[index];
        modelId = txRow.model_id || 'random_forest_smote';
    }

    if (!txRow) {
        aiText.innerHTML = '<p style="color:#ef4444;">Không tìm thấy dữ liệu giao dịch.</p>';
        spinnerEl.style.display = 'none';
        return;
    }

    // Build raw features payload
    const payload = {
        Time: txRow.time_val,
        Amount: txRow.amount,
        model_id: modelId,
    };
    // V features
    const vf = txRow.v_features || {};
    for (let i = 1; i <= 28; i++) {
        payload[`V${i}`] = vf[`V${i}`] || 0;
    }

    try {
        // 1. Get SHAP chart
        const res = await fetch('/explain-raw/', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
        });
        const data = await res.json();

        spinnerEl.style.display = 'none';

        if (!data.success) {
            aiText.innerHTML = `<p style="color:#ef4444;">Lỗi SHAP: ${data.error}</p>`;
            return;
        }

        // Show chart
        imgEl.src = data.image_base64;
        imgEl.classList.remove('hidden');

        // Store for re-analysis with different provider
        shapModalTxData = {
            top_features: data.top_features,
            verdict: txRow.verdict,
            probability: txRow.probability,
            amount: txRow.amount,
            model_used: lastBatchResults ? lastBatchResults.model_used : '',
        };

        // 2. Auto-trigger AI analysis
        runShapAIAnalysis(shapModalTxData);

    } catch (e) {
        spinnerEl.style.display = 'none';
        aiText.innerHTML = `<p style="color:#ef4444;">Lỗi kết nối: ${e.message}</p>`;
    }
}

async function runShapAIAnalysis(txData) {
    const aiText = document.getElementById('shapModalAiText');
    aiText.innerHTML = '<div class="spinner" style="width:24px;height:24px;margin:10px auto;"></div><p style="text-align:center;color:var(--text-muted);">AI đang đọc biểu đồ...</p>';

    try {
        const res = await fetch('/analyze-shap/', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                top_features: txData.top_features,
                verdict: txData.verdict,
                probability: txData.probability,
                amount: txData.amount,
                model_used: txData.model_used,
                provider: shapModalProvider,
            }),
        });
        const data = await res.json();
        aiText.innerText = data.analysis;
    } catch (e) {
        aiText.innerHTML = `<p style="color:#ef4444;">Không thể kết nối AI. Hãy thử đổi provider.</p>`;
    }
}
