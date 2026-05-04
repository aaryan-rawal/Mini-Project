/**
 * script.js — SpamShield v2  (clean)
 */

// ── DOM refs ───────────────────────────────────────────────────────────────
const msgInput     = document.getElementById("msgInput");
const analyseBtn   = document.getElementById("analyseBtn");
const clearBtn     = document.getElementById("clearBtn");
const btnText      = analyseBtn.querySelector(".btn-text");
const btnLoader    = analyseBtn.querySelector(".btn-loader");
const resultCard   = document.getElementById("resultCard");
const verdictBadge = document.getElementById("verdictBadge");
const confValue    = document.getElementById("confValue");
const confBar      = document.getElementById("confBar");
const kwBlock      = document.getElementById("keywordsBlock");
const kwTags       = document.getElementById("kwTags");
const historyBody  = document.getElementById("historyBody");
const themeToggle  = document.getElementById("themeToggle");
const toggleIcon   = document.getElementById("toggleIcon");
const toggleLabel  = document.getElementById("toggleLabel");


// ── Theme Toggle ───────────────────────────────────────────────────────────
const savedTheme = localStorage.getItem("theme") || "dark";
if (savedTheme === "light") applyLight();

themeToggle.addEventListener("click", () => {
  if (document.body.classList.contains("light")) {
    applyDark();
    localStorage.setItem("theme", "dark");
  } else {
    applyLight();
    localStorage.setItem("theme", "light");
  }
});

function applyLight() {
  document.body.classList.add("light");
  toggleIcon.textContent  = "☀️";
  toggleLabel.textContent = "Light Mode";
}

function applyDark() {
  document.body.classList.remove("light");
  toggleIcon.textContent  = "🌙";
  toggleLabel.textContent = "Dark Mode";
}


// ── Analyse ────────────────────────────────────────────────────────────────
analyseBtn.addEventListener("click", runPredict);
msgInput.addEventListener("keydown", e => {
  if ((e.ctrlKey || e.metaKey) && e.key === "Enter") runPredict();
});

async function runPredict() {
  const msg = msgInput.value.trim();
  if (!msg) { shake(msgInput); return; }

  setLoading(true);

  try {
    const res = await fetch("/predict", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ message: msg }),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      alert(err.error || "Something went wrong.");
      return;
    }

    const data = await res.json();
    showResult(data);
    await refreshHistory();

  } catch (err) {
    alert("Network error — make sure the Flask server is running.");
    console.error(err);
  } finally {
    setLoading(false);
  }
}


// ── Show Result ────────────────────────────────────────────────────────────
function showResult({ label, confidence, keywords }) {
  resultCard.hidden = false;
  resultCard.style.animation = "none";
  void resultCard.offsetWidth;
  resultCard.style.animation = "";

  const isSpam = label === "Spam";

  verdictBadge.textContent = isSpam ? "🚫 SPAM" : "✅ HAM";
  verdictBadge.className   = "verdict-badge " + (isSpam ? "spam" : "ham");

  confValue.textContent = confidence + "%";
  confValue.style.color = isSpam ? "var(--spam-red)" : "var(--ham-green)";

  confBar.className   = "conf-bar " + (isSpam ? "spam" : "ham");
  confBar.style.width = "0%";
  requestAnimationFrame(() => {
    requestAnimationFrame(() => { confBar.style.width = confidence + "%"; });
  });

  if (keywords && keywords.length > 0) {
    kwBlock.hidden   = false;
    kwTags.innerHTML = keywords.map(kw =>
      `<span class="kw-tag">${escHtml(kw)}</span>`
    ).join("");
  } else {
    kwBlock.hidden   = true;
    kwTags.innerHTML = "";
  }

  resultCard.scrollIntoView({ behavior: "smooth", block: "nearest" });
}


// ── History Refresh ────────────────────────────────────────────────────────
async function refreshHistory() {
  try {
    const res  = await fetch("/history");
    const rows = await res.json();
    if (!rows.length) return;

    historyBody.innerHTML = rows.map(row => {
      const isSpam   = row.result === "Spam";
      const msgShort = row.message.length > 60
        ? row.message.slice(0, 60) + "…"
        : row.message;
      return `
        <tr>
          <td class="td-time">${row.timestamp}</td>
          <td class="td-msg">${escHtml(msgShort)}</td>
          <td><span class="pill pill-${isSpam ? "spam" : "ham"}">${row.result}</span></td>
          <td class="td-conf">${row.confidence}</td>
          <td class="td-kw">${row.keywords}</td>
        </tr>`;
    }).join("");
  } catch (e) {
    console.warn("History refresh failed:", e);
  }
}


// ── Quick-test buttons ─────────────────────────────────────────────────────
document.querySelectorAll(".qt-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    msgInput.value = btn.dataset.msg;
    msgInput.focus();
    runPredict();
  });
});


// ── Clear ──────────────────────────────────────────────────────────────────
clearBtn.addEventListener("click", () => {
  msgInput.value    = "";
  resultCard.hidden = true;
  msgInput.focus();
});


// ── Helpers ────────────────────────────────────────────────────────────────
function setLoading(on) {
  analyseBtn.disabled = on;
  btnText.hidden      = on;
  btnLoader.hidden    = !on;
}

function shake(el) {
  el.style.animation = "none";
  void el.offsetWidth;
  el.style.animation = "shake 0.4s ease";
  el.addEventListener("animationend", () => { el.style.animation = ""; }, { once: true });
}

function escHtml(str) {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

const shakeStyle = document.createElement("style");
shakeStyle.textContent = `
  @keyframes shake {
    0%,100% { transform: translateX(0); }
    20%     { transform: translateX(-6px); }
    40%     { transform: translateX(6px); }
    60%     { transform: translateX(-4px); }
    80%     { transform: translateX(4px); }
  }
`;
document.head.appendChild(shakeStyle);
