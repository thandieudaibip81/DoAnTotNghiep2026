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
        btn.innerHTML = '<i class="fas fa-robot"></i> Phân tích bằng AI';
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

    list.innerHTML = filtered.map(item => {
        let cls = 'status-safe';
        if (item.verdict === 'Gian lận') cls = 'status-fraud';
        else if (item.verdict === 'Nghi ngờ') cls = 'status-suspicious';

        return `
            <div class="history-item ${item.id === currentSelectedId ? 'active' : ''}" data-id="${item.id}" onclick="selectHistoryItem(${item.id})">
                <div class="item-header">
                    <span>#${item.id}</span>
                    <span>${item.timestamp}</span>
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
            history.map(h => `<option value="${h.id}">#${h.id} - $${h.amount}</option>`).join('');
        sel.value = cur;
    }
}

async function selectHistoryItem(id) {
    if (currentSelectedId === id) { resetUI(); return; }
    currentSelectedId = id;
    document.querySelectorAll('.history-item').forEach(el => el.classList.remove('active'));
    const target = document.querySelector(`.history-item[data-id="${id}"]`);
    if (target) target.classList.add('active');

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
// INIT
// ══════════════════════════════════════════════════

document.addEventListener('DOMContentLoaded', () => {
    loadModels();
    loadHistory();
});
