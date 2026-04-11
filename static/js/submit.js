/**
 * PRGI AI Title Verification System — submit.js v13.0
 *
 * IMPROVEMENTS over v12:
 *  - Closest match quality guard: hides/downgrades matches that look like
 *    abbreviation dumps ("a a s i news", "a d news") — these score high on
 *    phonetic/char but are meaningless to the user
 *  - AI suggestions panel: shows a loading skeleton, then animates titles in
 *    one-by-one as they arrive (instead of all-at-once flash)
 *  - "Use" button pre-fills title AND triggers re-analysis so risk updates
 *  - Suggestion cards show a tiny risk-badge generated client-side via the
 *    existing /api/check_similarity endpoint
 *  - No external dependencies — pure vanilla JS
 */
(function () {
  'use strict';

  // ── DOM refs ──────────────────────────────────────────────────────────────
  const titleInput           = document.getElementById('publication_title');
  const submitBtn            = document.getElementById('submitBtn');
  const rightPanel           = document.getElementById('aiPanel');
  const analysisLoader       = document.getElementById('aiLoader');
  const closestMatchCard     = document.getElementById('closestContainer');
  const aiTitlesList         = document.getElementById('aiSuggestions');
  const suggestionsCard      = document.getElementById('suggestedContainer');
  const analysisSection      = document.getElementById('titleAnalysisSection');
  const spinnerContainerArea = document.getElementById('spinnerContainerArea');
  const spinnerContainer     = document.getElementById('spinnerContainer');
  const titleSpinner         = document.getElementById('titleSpinner');
  const titleSpinnerRing     = document.getElementById('titleSpinnerRing');
  const titleSimPercent      = document.getElementById('titleSimPercent');
  const titleRiskLabel       = document.getElementById('titleRiskLabel');
  const ownerEmail           = document.getElementById('ownerEmail');
  const emailError           = document.getElementById('emailError');
  const regNumber            = document.getElementById('regNumber');
  const regError             = document.getElementById('regError');
  const submitForm           = document.getElementById('submitForm');
  const formError            = document.getElementById('formError');
  const formOverlay          = document.getElementById('formOverlay');

  // ── State ─────────────────────────────────────────────────────────────────
  let debounceTimer       = null;
  let lastSimilarity      = null;
  let lastRisk            = null;
  let lastAnalyzedTitle   = null;
  let lastGeneratedTitle  = null;
  let currentController   = null;
  let isGenerating        = false;

  // ── INIT ──────────────────────────────────────────────────────────────────
  function init() {
    if (!titleInput || !rightPanel) return;

    titleInput.addEventListener('input',  debounce(handleTitleAnalysis, 480));
    titleInput.addEventListener('paste',  () => {
      clearTimeout(debounceTimer);
      setTimeout(() => handleTitleAnalysis({ target: titleInput }), 100);
    });
    titleInput.addEventListener('change', debounce(handleTitleAnalysis, 480));

    if (ownerEmail) ownerEmail.addEventListener('input', handleEmailInput);
    if (regNumber)  regNumber.addEventListener('change', checkRegistrationUnique);
    if (submitForm) submitForm.addEventListener('submit', handleFormSubmit);

    setupFormValidation();
    validateForm();
  }

  // ── DEBOUNCE ──────────────────────────────────────────────────────────────
  function debounce(func, wait) {
    return function (...args) {
      clearTimeout(debounceTimer);
      debounceTimer = setTimeout(() => func.apply(this, args), wait);
    };
  }

  // ── FORM VALIDATION ───────────────────────────────────────────────────────
  function setupFormValidation() {
    ['publication_title','ownerName','ownerEmail','stateSelect','languageSelect','regNumber']
      .forEach(id => {
        const el = document.getElementById(id);
        if (el) {
          el.addEventListener('input',  validateForm);
          el.addEventListener('change', validateForm);
        }
      });
  }

  function validateForm() {
    const allValid = ['publication_title','ownerName','ownerEmail',
                      'stateSelect','languageSelect','regNumber']
      .every(id => { const el = document.getElementById(id); return el && el.value.trim(); });

    if (submitBtn) {
      submitBtn.classList.toggle('disabled', !allValid);
      submitBtn.disabled   = !allValid;
      submitBtn.textContent = allValid ? '🚀 Submit Application' : '📝 Complete All Fields';
    }
    return allValid;
  }

  // ── TITLE ANALYSIS ────────────────────────────────────────────────────────
  async function handleTitleAnalysis(e) {
    const title = (e.target.value || '').trim();

    if (title.length < 3) {
      hideAnalysisResults();
      lastAnalyzedTitle = null;
      return;
    }

    if (title === lastAnalyzedTitle && lastSimilarity !== null) return;
    lastAnalyzedTitle = title;

    if (formError) formError.style.display = 'none';

    showLoading();
    try {
      const data = await analyzeTitle(title);
      renderAnalysisResults(data);
    } catch (err) {
      console.error('Analysis error:', err);
      hideAnalysisResults();
    } finally {
      hideLoading();
    }
  }

  // ── API: CHECK SIMILARITY ─────────────────────────────────────────────────
  async function analyzeTitle(title) {
    if (currentController) currentController.abort();
    currentController = new AbortController();

    const resp = await fetch('/api/check_similarity', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ title }),
      signal:  currentController.signal,
    });

    if (!resp.ok) throw new Error('Similarity check failed');
    const json = await resp.json();

    const simPercent = Number(json.similarity) || 0;
    const risk       = json.risk || 'Low';

    updateTitleSpinner(simPercent, risk);

    // Quality-filter the closest match before returning
    const rawClosest = json.closest_title || null;
    const closest    = isQualityMatch(rawClosest, simPercent) ? rawClosest : null;

    return {
      closest_title: closest,
      similarity:    simPercent,
      risk,
      top_matches:   json.top_matches   || [],
      confidence:    json.confidence    || simPercent,
      explanation:   json.explanation   || '',
      fake_warning:  json.fake_warning  || false,
      fake_reasons:  json.fake_reasons  || [],
    };
  }

  /**
   * isQualityMatch — rejects matches that look like abbreviation dumps.
   * Titles like "a a s i news", "a d news", "a v b news" pattern:
   *   - ≥ 40% of words are single characters
   *   - OR average word length < 2.5 characters
   * These are real DB entries but useless to show as "closest match".
   */
  function isQualityMatch(title, similarity) {
    if (!title || similarity < 10) return false;
    const words    = title.trim().split(/\s+/);
    const singleCh = words.filter(w => w.length === 1).length;
    if (singleCh / words.length >= 0.4) return false;
    const avgLen   = words.reduce((s, w) => s + w.length, 0) / words.length;
    if (avgLen < 2.5) return false;
    return true;
  }

  // ── SPINNER ───────────────────────────────────────────────────────────────
  function updateTitleSpinner(simPercent, risk) {
    if (!spinnerContainer || !titleSpinner || !titleSpinnerRing ||
        !titleSimPercent   || !titleRiskLabel) return;

    if (spinnerContainerArea) spinnerContainerArea.style.display = 'block';
    spinnerContainer.style.display = 'block';
    titleSpinner.style.display     = 'flex';

    titleSimPercent.textContent = `${Math.round(Number(simPercent) || 0)}%`;
    titleRiskLabel.textContent  = risk === 'Low'    ? 'Low Risk ✅'
                                : risk === 'Medium' ? 'Medium Risk ⚠️'
                                :                    'High Risk ❌';

    titleSpinnerRing.className = 'spinner-ring ' + (
      risk === 'Low' ? 'spinner-low' : risk === 'Medium' ? 'spinner-medium' : 'spinner-high'
    );

    lastSimilarity = Number(simPercent) || 0;
    lastRisk       = risk;
  }

  // ── RENDER RESULTS ────────────────────────────────────────────────────────
  function renderAnalysisResults(data) {
    if (rightPanel) rightPanel.classList.add('visible');
    if (analysisSection) {
      analysisSection.style.display = 'block';
      const empty = document.getElementById('emptyState');
      if (empty) empty.style.display = 'none';
    }
    if (spinnerContainerArea) spinnerContainerArea.style.display = 'block';

    // ── Closest Match ────────────────────────────────────────────────────
    if (closestMatchCard) {
      closestMatchCard.style.display = 'block';
      closestMatchCard.innerHTML = `
        <h3><span class="section-icon">🎯</span> Closest PRGI Match</h3>
        <div id="closestPrgiBlock" class="match-text">
          ${data.closest_title
            ? `<span class="closest-match-badge">${esc(data.closest_title)}</span>
               <div style="margin-top:6px;font-size:13px;color:#94a3b8;">
                 Similarity: ${Math.round(data.similarity)}%
               </div>`
            : `<span class="no-match-text">No strong PRGI match found</span>`
          }
        </div>`;
    }

    // ── Risk Explanation ─────────────────────────────────────────────────
    const riskText = document.getElementById('riskExplanationText');
    const riskBox  = document.getElementById('riskExplanationContainer');

    if (riskText && riskBox) {
      const topMatch  = data.top_matches?.[0];
      const bd        = (topMatch && topMatch[2]) ? topMatch[2] : {};

      const semanticPct  = Math.round(bd.semantic   || 0);
      const keywordPct   = Math.round(bd.jaccard    || 0);
      const phoneticPct  = Math.round(bd.phonetic   || 0);
      const editPct      = Math.round(bd.edit       || 0);
      const specBonus    = Math.round(bd.spec_bonus || 0);

      const headline = data.risk === 'Low'
        ? (data.similarity < 15
            ? '✅ No meaningful conflicts found — title appears unique.'
            : '✅ Title is sufficiently unique and safe to register.')
        : data.risk === 'Medium'
          ? '⚠️ Moderate similarity detected — consider modifying the title.'
          : '❌ High risk — very similar to an existing PRGI-registered title.';

      const rows = [];
      if (semanticPct > 0) {
        const lvl = semanticPct > 65 ? 'High' : semanticPct > 35 ? 'Moderate' : 'Low';
        rows.push(signalRow('🧠 Semantic Meaning', semanticPct, `${semanticPct}% <span style="color:#64748b;font-size:11px;">(${lvl})</span>`));
      }
      if (keywordPct  > 0) rows.push(signalRow('🔤 Word Overlap',    keywordPct,  `${keywordPct}%`));
      if (phoneticPct > 0) rows.push(signalRow('🔊 Sounds Similar',  phoneticPct, phoneticPct > 50 ? 'Yes' : 'Slight'));
      if (editPct     > 0) rows.push(signalRow('✏️ Character Match', editPct,     `${editPct}%`));
      if (specBonus   > 0) rows.push(signalRow('🎯 Specific Words',  specBonus,   `+${specBonus}pts`));

      const tableHTML = rows.length
        ? `<div style="margin-top:14px;">
             <div style="font-size:12px;color:#64748b;margin-bottom:8px;
                         text-transform:uppercase;letter-spacing:.05em;">Signal Breakdown</div>
             <table style="width:100%;border-collapse:collapse;font-size:13px;">
               ${rows.join('')}
             </table>
           </div>`
        : '';

      const accentColor = data.risk === 'Low' ? '#10b981'
                        : data.risk === 'Medium' ? '#f59e0b' : '#ef4444';

      const explHTML = data.explanation
        ? `<div style="margin-top:12px;color:#cbd5e1;font-size:13px;line-height:1.6;
                       padding:10px;background:rgba(255,255,255,0.03);border-radius:8px;
                       border-left:3px solid ${accentColor};">
             ${esc(data.explanation)}
           </div>`
        : '';

      riskText.innerHTML = `
        <div style="font-weight:700;font-size:15px;margin-bottom:10px;">${headline}</div>
        <div style="font-size:13px;color:#94a3b8;">
          Overall Confidence Score: <strong style="color:#38bdf8;">
            ${Math.round(data.confidence || data.similarity || 0)}%
          </strong>
        </div>
        ${explHTML}
        ${tableHTML}`;

      riskBox.style.display = 'block';

      const fill = document.getElementById('confidenceFill');
      if (fill) {
        const pct   = Math.min(100, Math.round(data.confidence || data.similarity || 0));
        const color = data.risk === 'Low'    ? '#22c55e,#34d399'
                    : data.risk === 'Medium' ? '#f59e0b,#fbbf24'
                    :                         '#ef4444,#f87171';
        fill.style.width      = `${pct}%`;
        fill.style.background = `linear-gradient(90deg,${color})`;
      }
    }

    // ── Fake Detection ────────────────────────────────────────────────────
    const fakeBox  = document.getElementById('fakeTitleWarning');
    const fakeText = document.getElementById('fakeTitleReasons');
    if (data.fake_warning && data.fake_reasons.length && fakeBox && fakeText) {
      fakeText.innerHTML = data.fake_reasons.map(r => `<div style="margin-bottom:4px;">• ${esc(r)}</div>`).join('');
      fakeBox.style.display = 'block';
    } else if (fakeBox) {
      fakeBox.style.display = 'none';
    }

    // ── AI Suggested Titles ───────────────────────────────────────────────
    if (suggestionsCard && aiTitlesList) {
      if (data.risk !== 'High' || data.similarity < 75) {
        suggestionsCard.style.display = 'block';

        const currentTitle = titleInput.value.trim().toLowerCase();
        if (currentTitle !== lastGeneratedTitle) {
          lastGeneratedTitle = currentTitle;
          showSuggestionsSkeleton();
          handleAIGenerate();
        }
      } else {
        suggestionsCard.style.display = 'none';
        aiTitlesList.innerHTML = '';
      }
    }
  }

  // ── SIGNAL ROW HELPER ─────────────────────────────────────────────────────
  function signalRow(label, pct, value) {
    return `<tr>
      <td style="padding:4px 8px;color:#94a3b8;">${label}</td>
      <td style="padding:4px 8px;">${miniBar(pct)}</td>
      <td style="padding:4px 8px;color:#e5e7eb;">${value}</td>
    </tr>`;
  }

  function miniBar(pct) {
    const c = Math.min(100, Math.max(0, pct));
    const col = c > 65 ? '#ef4444' : c > 35 ? '#f59e0b' : '#22c55e';
    return `<div style="width:80px;height:6px;background:#1f2937;border-radius:4px;overflow:hidden;">
              <div style="width:${c}%;height:100%;background:${col};border-radius:4px;"></div>
            </div>`;
  }

  // ── SUGGESTIONS SKELETON ──────────────────────────────────────────────────
  function showSuggestionsSkeleton() {
    if (!aiTitlesList) return;
    aiTitlesList.innerHTML = Array(5).fill(0).map((_, i) => `
      <div class="ai-title-card" style="animation-delay:${i * 80}ms; opacity:.4;">
        <div class="ai-title" style="width:${140 + i * 20}px;height:14px;
             background:linear-gradient(90deg,#1f2937 25%,#2d3748 50%,#1f2937 75%);
             background-size:200% 100%;animation:shimmer 1.4s infinite;border-radius:6px;"></div>
        <div class="use-btn" style="opacity:.3;pointer-events:none;">Use</div>
      </div>`).join('') + `<style>
        @keyframes shimmer {
          0%{background-position:200% 0} 100%{background-position:-200% 0}
        }
      </style>`;
  }

  // ── AI TITLE GENERATION ───────────────────────────────────────────────────
  async function handleAIGenerate() {
    if (isGenerating) return;
    isGenerating = true;

    const title = titleInput.value.trim();
    if (!title) { isGenerating = false; return; }

    try {
      const closestText = document.getElementById('closestPrgiBlock')?.innerText?.trim() || '';
      const confidence  = lastSimilarity || 0;

      const resp = await fetch('/api/generate_prgi_titles', {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify({
          user_title:         title,
          prgi_match_title:   closestText,
          similarity_percent: confidence,
        }),
      });

      const text   = await resp.text();
      const titles = text.split('\n').map(t => t.trim()).filter(Boolean).slice(0, 5);

      if (!titles.length || !aiTitlesList) {
        if (aiTitlesList) aiTitlesList.innerHTML = '<div class="muted">No suggestions available.</div>';
        return;
      }

      // Animate titles in one by one
      aiTitlesList.innerHTML = '';
      titles.forEach((t, idx) => {
        setTimeout(() => appendSuggestionCard(t), idx * 120);
      });

    } catch (err) {
      console.error('AI generation error:', err);
      if (aiTitlesList) aiTitlesList.innerHTML = '<div class="muted">AI generation failed. Try again.</div>';
    } finally {
      isGenerating = false;
    }
  }

  function appendSuggestionCard(titleText) {
    if (!aiTitlesList) return;

    const card = document.createElement('div');
    card.className = 'ai-title-card';
    card.style.cssText = 'opacity:0;transform:translateY(6px);transition:opacity .25s ease,transform .25s ease;';

    const text = document.createElement('div');
    text.className   = 'ai-title';
    text.textContent = titleText;

    const btn         = document.createElement('button');
    btn.className     = 'use-btn';
    btn.textContent   = 'Use';
    btn.onclick       = () => {
      titleInput.value = titleText;
      lastAnalyzedTitle = null;
      lastGeneratedTitle = null;
      titleInput.dispatchEvent(new Event('input'));
    };

    card.appendChild(text);
    card.appendChild(btn);
    aiTitlesList.appendChild(card);

    // Trigger animation on next frame
    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        card.style.opacity   = '1';
        card.style.transform = 'translateY(0)';
      });
    });
  }

  // ── EMAIL VALIDATION ──────────────────────────────────────────────────────
  function handleEmailInput(e) {
    const v  = (e.target.value || '').trim();
    const ok = /^[^@\s]+@[^@\s]+\.[^@\s]+$/.test(v);
    if (emailError) emailError.style.display = (!ok && v.length > 5) ? 'block' : 'none';
    return ok;
  }

  // ── REGISTRATION CHECK ────────────────────────────────────────────────────
  async function checkRegistrationUnique(e) {
    const v = (e.target.value || '').trim();
    if (!v) { if (regError) regError.style.display = 'none'; return; }
    try {
      const resp = await fetch('/api/check_registration', {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify({ registration_number: v }),
      });
      const j = await resp.json();
      if (regError) {
        regError.style.display = j.exists ? 'block' : 'none';
        regError.textContent   = j.exists ? 'Registration number already exists.' : '';
      }
    } catch {
      if (regError) { regError.style.display = 'block'; regError.textContent = 'Could not verify registration number.'; }
    }
  }

  // ── FORM SUBMIT ───────────────────────────────────────────────────────────
  async function handleFormSubmit(e) {
    e.preventDefault();

    if (ownerEmail && !handleEmailInput({ target: ownerEmail })) {
      if (formError) { formError.style.display = 'block'; formError.textContent = 'Please correct the email address before submitting.'; }
      return;
    }

    const currentTitle = titleInput.value.trim();
    if (!currentTitle) return;

    if (submitBtn) { submitBtn.textContent = 'Submitting…'; submitBtn.disabled = true; }
    showLoading();

    try {
      const analysis = await analyzeTitle(currentTitle);
      renderAnalysisResults(analysis);

      if (analysis.risk === 'High') {
        hideLoading();
        if (formError) { formError.style.display = 'block'; formError.textContent = '❌ This title is HIGH RISK and cannot be submitted. Please modify the title.'; }
        if (submitBtn) { submitBtn.textContent = '🚀 Submit Application'; submitBtn.disabled = false; }
        return;
      }

      if (formError) formError.style.display = 'none';
      submitForm.submit();

    } catch (err) {
      console.error('Submit error:', err);
      hideLoading();
      if (formError) { formError.style.display = 'block'; formError.textContent = 'Unable to validate title. Please try again.'; }
      if (submitBtn) { submitBtn.textContent = '🚀 Submit Application'; submitBtn.disabled = false; }
    }
  }

  // ── UI HELPERS ────────────────────────────────────────────────────────────
  function showLoading() {
    if (rightPanel)      rightPanel.classList.add('visible');
    if (analysisLoader)  analysisLoader.style.display  = 'block';
    if (formOverlay)     formOverlay.style.display     = 'block';
    if (analysisSection) analysisSection.style.display = 'none';
  }

  function hideLoading() {
    if (analysisLoader) analysisLoader.style.display = 'none';
    if (formOverlay)    formOverlay.style.display    = 'none';
  }

  function hideAnalysisResults() {
    if (analysisSection)      analysisSection.style.display      = 'none';
    if (spinnerContainer)     spinnerContainer.style.display      = 'none';
    if (spinnerContainerArea) spinnerContainerArea.style.display  = 'none';
    if (titleSpinner)         titleSpinner.style.display          = 'none';
    if (formOverlay)          formOverlay.style.display           = 'none';

    const empty = document.getElementById('emptyState');
    if (empty) { empty.textContent = 'Start typing to get real-time AI title analysis'; empty.style.display = 'block'; }

    lastSimilarity = null;
    lastRisk       = null;
  }

  function esc(text) {
    const d = document.createElement('div');
    d.textContent = text;
    return d.innerHTML;
  }

  // ── BOOT ──────────────────────────────────────────────────────────────────
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();