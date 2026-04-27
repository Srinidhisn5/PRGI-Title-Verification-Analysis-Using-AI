/**
 * PRGI AI Title Verification System — submit.js v14.1
 *
 * Fixes in v14.1:
 *  - Low risk (< 30%) → hides suggestions panel entirely, shows only rewrite button
 *  - Medium/High risk → shows full 5 AI suggestions as before
 *  - Fake detection "Title too short" replaced with proper AI message
 *  - NLP fake reasons no longer shown (AI handles fake detection now)
 *  - Suggestions header badge shows "Groq AI" or "Gemini AI" correctly
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
  let debounceTimer      = null;
  let lastSimilarity     = null;
  let lastRisk           = null;
  let lastAnalyzedTitle  = null;
  let lastGeneratedTitle = null;
  let currentController  = null;
  let isGenerating       = false;
  let lastAnalysisData   = null;

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
      submitBtn.disabled    = !allValid;
      submitBtn.textContent = allValid ? '🚀 Submit Application' : '📝 Complete All Fields';
    }
    return allValid;
  }

  // ── TITLE ANALYSIS ENTRY POINT ────────────────────────────────────────────
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
      // Step 1: NLP similarity (fast — SBERT engine)
      const simData = await analyzeTitle(title);

      // Step 2: Render NLP results immediately
      renderNLPResults(simData);

      // Step 3: Enhance with AI (Groq calls — parallel, non-blocking)
      enhanceWithAI(title, simData);

    } catch (err) {
      console.error('Analysis error:', err);
      hideAnalysisResults();
    } finally {
      hideLoading();
    }
  }

  // ── API: NLP Similarity (existing engine) ─────────────────────────────────
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

    const rawClosest = json.closest_title || null;
    const closest    = isQualityMatch(rawClosest, simPercent) ? rawClosest : null;

    return {
      closest_title: closest,
      similarity:    simPercent,
      risk,
      top_matches:   json.top_matches   || [],
      confidence:    json.confidence    || simPercent,
      explanation:   json.explanation   || '',
      // Note: NLP fake_warning intentionally NOT used — AI handles fake detection
    };
  }

  // ── QUALITY GUARD ─────────────────────────────────────────────────────────
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

  // ── RENDER NLP RESULTS ────────────────────────────────────────────────────
  function renderNLPResults(data) {
    lastAnalysisData = data;

    if (rightPanel) rightPanel.classList.add('visible');
    if (analysisSection) {
      analysisSection.style.display = 'block';
      const empty = document.getElementById('emptyState');
      if (empty) empty.style.display = 'none';
    }
    if (spinnerContainerArea) spinnerContainerArea.style.display = 'block';

    // ── Closest Match ─────────────────────────────────────────────────────
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

    // ── Risk Explanation (placeholder while Groq loads) ───────────────────
    const riskBox  = document.getElementById('riskExplanationContainer');
    const riskText = document.getElementById('riskExplanationText');

    if (riskBox && riskText) {
      const topMatch = data.top_matches?.[0];
      const bd       = (topMatch && topMatch[2]) ? topMatch[2] : {};

      const semanticPct  = Math.round(bd.semantic   || 0);
      const keywordPct   = Math.round(bd.jaccard    || 0);
      const phoneticPct  = Math.round(bd.phonetic   || 0);
      const editPct      = Math.round(bd.edit       || 0);
      const specBonus    = Math.round(bd.spec_bonus || 0);

      const accentColor = data.risk === 'Low'    ? '#10b981'
                        : data.risk === 'Medium' ? '#f59e0b' : '#ef4444';

      const rows = [];
      if (semanticPct > 0) {
        const lvl = semanticPct > 65 ? 'High' : semanticPct > 35 ? 'Moderate' : 'Low';
        rows.push(signalRow('🧠 Semantic Meaning', semanticPct,
          `${semanticPct}% <span style="color:#64748b;font-size:11px;">(${lvl})</span>`));
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
           </div>` : '';

      riskText.innerHTML = `
        <div id="aiExplanationBlock"
             style="margin-bottom:12px;padding:10px;background:rgba(255,255,255,0.03);
                    border-radius:8px;border-left:3px solid ${accentColor};">
          <div style="display:flex;align-items:center;gap:8px;color:#64748b;font-size:13px;">
            <div class="spinner-ring" style="width:14px;height:14px;border-width:2px;
                 border-top-color:${accentColor};"></div>
            🤖 AI is analyzing your title...
          </div>
        </div>
        <div style="font-size:13px;color:#94a3b8;">
          Overall Confidence Score: <strong style="color:#38bdf8;">
            ${Math.round(data.confidence || data.similarity || 0)}%
          </strong>
        </div>
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

    // ── Fake Detection — hide NLP version, AI will handle it ─────────────
    const fakeBox = document.getElementById('fakeTitleWarning');
    if (fakeBox) fakeBox.style.display = 'none';

    // ── Suggestions — LOW RISK: hide suggestions, show only rewrite button
    //                  MEDIUM/HIGH: show full suggestions panel ─────────────
    if (suggestionsCard && aiTitlesList) {
      const isLowRisk = data.risk === 'Low' || data.similarity < 30;

      if (isLowRisk) {
        // Low risk: no suggestions needed — show minimal rewrite option instead
        suggestionsCard.style.display = 'block';
        const header = suggestionsCard.querySelector('h3');
        if (header) {
          header.innerHTML = `<span class="section-icon">✨</span> AI Title Tools`;
        }
        aiTitlesList.innerHTML = `
          <div style="padding:12px;background:rgba(16,185,129,0.08);border:1px solid rgba(16,185,129,0.2);
                      border-radius:10px;margin-bottom:12px;">
            <div style="font-size:13px;color:#10b981;font-weight:600;margin-bottom:4px;">
              ✅ Your title looks unique
            </div>
            <div style="font-size:12px;color:#64748b;">
              No similar PRGI titles found. You can proceed with registration or use AI to explore alternatives.
            </div>
          </div>`;

        // Still show rewrite button for exploration
        const currentTitle = titleInput.value.trim();
        setTimeout(() => appendRewriteButton(currentTitle, ''), 200);

      } else {
        // Medium/High risk: show full suggestions
        suggestionsCard.style.display = 'block';
        const header = suggestionsCard.querySelector('h3');
        if (header) {
          header.innerHTML = `<span class="section-icon">🪄</span> AI Suggested Titles
            <span style="font-size:11px;font-weight:700;color:#64748b;
              background:rgba(100,116,139,0.1);padding:2px 8px;border-radius:12px;margin-left:8px;">
              Loading...</span>`;
        }
        showSuggestionsSkeleton();

        const currentTitle = titleInput.value.trim().toLowerCase();
        if (currentTitle !== lastGeneratedTitle) {
          lastGeneratedTitle = currentTitle;
          fetchAISuggestions(
            titleInput.value.trim(),
            data.closest_title || '',
            data.similarity
          );
        }
      }
    }
  }

  // ── AI ENHANCEMENT ────────────────────────────────────────────────────────
  async function enhanceWithAI(title, simData) {
    const closest   = simData.closest_title || '';
    const topMatch  = simData.top_matches?.[0];
    const breakdown = (topMatch && topMatch[2]) ? topMatch[2] : {};

    // Run AI explanation + fake detect together
    fetchAIExplanation(title, closest, simData.similarity, simData.risk, breakdown);
  }

  // ── AI: Full Analysis (explanation + fake detect) ─────────────────────────
  async function fetchAIExplanation(title, closest, simPct, risk, breakdown) {
    try {
      const resp = await fetch('/api/ai/full_analysis', {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify({
          user_title:         title,
          closest_match:      closest,
          similarity_percent: simPct,
          risk_level:         risk,
          breakdown:          breakdown,
        }),
      });

      if (!resp.ok) throw new Error('AI analysis call failed');
      const data = await resp.json();
      if (!data.success) throw new Error(data.error || 'AI analysis failed');

      updateAIExplanation(data.explanation, risk);

      // Only show AI fake detection if actually suspicious
      if (data.is_fake && data.fake_reasons && data.fake_reasons.length > 0) {
        updateAIFakeDetection(data.fake_reasons, data.fake_confidence);
      }

    } catch (err) {
      console.warn('AI explanation failed, using NLP fallback:', err.message);
      clearAIExplanationSpinner(risk, simPct);
    }
  }

  function updateAIExplanation(explanation, risk) {
    const block = document.getElementById('aiExplanationBlock');
    if (!block) return;

    const accentColor = risk === 'Low'    ? '#10b981'
                      : risk === 'Medium' ? '#f59e0b' : '#ef4444';

    const aiLabel = `<span style="display:inline-flex;align-items:center;gap:4px;
      font-size:11px;font-weight:700;color:#7c5cff;letter-spacing:.04em;
      background:rgba(124,92,255,0.1);padding:2px 8px;border-radius:12px;
      margin-bottom:8px;">✨ AI Generated</span>`;

    block.style.borderLeftColor = accentColor;
    block.innerHTML = `
      ${aiLabel}
      <div style="color:#cbd5e1;font-size:13px;line-height:1.65;">${esc(explanation)}</div>`;
  }

  function clearAIExplanationSpinner(risk, simPct) {
    const block = document.getElementById('aiExplanationBlock');
    if (!block) return;
    const accentColor = risk === 'Low' ? '#10b981' : risk === 'Medium' ? '#f59e0b' : '#ef4444';
    block.style.borderLeftColor = accentColor;
    const msg = risk === 'Low'    ? '✅ Title appears sufficiently unique.'
              : risk === 'Medium' ? '⚠️ Moderate similarity detected — consider modifying.'
              :                    '❌ High risk — very similar to an existing registered title.';
    block.innerHTML = `<div style="color:#cbd5e1;font-size:13px;">${msg}</div>`;
  }

  function updateAIFakeDetection(reasons, confidence) {
    const fakeBox  = document.getElementById('fakeTitleWarning');
    const fakeText = document.getElementById('fakeTitleReasons');
    if (!fakeBox || !fakeText) return;

    const aiLabel = `<span style="font-size:11px;font-weight:700;color:#7c5cff;
      background:rgba(124,92,255,0.1);padding:2px 8px;border-radius:12px;margin-left:8px;">
      ✨ AI</span>`;

    fakeBox.style.display = 'block';
    fakeText.innerHTML = `
      ${reasons.map(r => `<div style="margin-bottom:6px;color:#fbbf24;">⚠️ ${esc(r)}</div>`).join('')}
      <div style="margin-top:8px;font-size:12px;color:#64748b;">
        AI Confidence: ${esc(confidence)} ${aiLabel}
      </div>`;
  }

  // ── AI: Suggestions ───────────────────────────────────────────────────────
  async function fetchAISuggestions(title, closest, simPct) {
    if (isGenerating) return;
    isGenerating = true;

    try {
      const resp = await fetch('/api/ai/suggestions', {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify({
          user_title:         title,
          closest_match:      closest,
          similarity_percent: simPct,
          language:           getSelectedLanguage(),
        }),
      });

      if (!resp.ok) throw new Error('AI suggestions failed');
      const data = await resp.json();
      const titles = (data.suggestions || []).slice(0, 5);

      if (!titles.length || !aiTitlesList) {
        if (aiTitlesList) aiTitlesList.innerHTML = '<div class="muted">No suggestions available.</div>';
        return;
      }

      aiTitlesList.innerHTML = '';

      // Update header badge with actual source
      const header = suggestionsCard?.querySelector('h3');
      if (header) {
        const sourceLabel = data.source === 'groq' ? 'Groq AI' : 'Gemini AI';
        header.innerHTML = `<span class="section-icon">🪄</span> AI Suggested Titles
          <span style="font-size:11px;font-weight:700;color:#7c5cff;
            background:rgba(124,92,255,0.1);padding:2px 8px;border-radius:12px;margin-left:8px;">
            ✨ ${sourceLabel}</span>`;
      }

      titles.forEach((t, idx) => {
        setTimeout(() => appendSuggestionCard(t), idx * 120);
      });

      // Rewrite button after suggestions
      setTimeout(() => appendRewriteButton(title, closest), titles.length * 120 + 100);

    } catch (err) {
      console.error('AI suggestion error:', err);
      // Fallback to old learning engine
      fetchLegacySuggestions(title, closest, simPct);
    } finally {
      isGenerating = false;
    }
  }

  // ── Fallback: old learning engine ─────────────────────────────────────────
  async function fetchLegacySuggestions(title, closest, simPct) {
    try {
      const resp = await fetch('/api/generate_prgi_titles', {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify({
          user_title:         title,
          prgi_match_title:   closest,
          similarity_percent: simPct,
        }),
      });
      const text   = await resp.text();
      const titles = text.split('\n').map(t => t.trim()).filter(Boolean).slice(0, 5);
      if (!aiTitlesList) return;
      aiTitlesList.innerHTML = '';
      titles.forEach((t, idx) => setTimeout(() => appendSuggestionCard(t), idx * 120));
      setTimeout(() => appendRewriteButton(title, closest), titles.length * 120 + 100);
    } catch (err) {
      if (aiTitlesList) aiTitlesList.innerHTML = '<div class="muted">Suggestions unavailable.</div>';
    }
  }

  // ── Suggestion Card ───────────────────────────────────────────────────────
  function appendSuggestionCard(titleText) {
    if (!aiTitlesList) return;
    const card = document.createElement('div');
    card.className = 'ai-title-card';
    card.style.cssText = 'opacity:0;transform:translateY(6px);transition:opacity .25s ease,transform .25s ease;';

    const text       = document.createElement('div');
    text.className   = 'ai-title';
    text.textContent = titleText;

    const btn         = document.createElement('button');
    btn.className     = 'use-btn';
    btn.textContent   = 'Use';
    btn.onclick       = () => {
      titleInput.value   = titleText;
      lastAnalyzedTitle  = null;
      lastGeneratedTitle = null;
      titleInput.dispatchEvent(new Event('input'));
    };

    card.appendChild(text);
    card.appendChild(btn);
    aiTitlesList.appendChild(card);

    requestAnimationFrame(() => requestAnimationFrame(() => {
      card.style.opacity   = '1';
      card.style.transform = 'translateY(0)';
    }));
  }

  // ── Feature 3: Make My Title Unique button ────────────────────────────────
  function appendRewriteButton(originalTitle, closestMatch) {
    if (!aiTitlesList) return;

    const wrapper = document.createElement('div');
    wrapper.style.cssText = 'margin-top:14px;padding-top:12px;border-top:1px solid #1f2937;';

    const btn = document.createElement('button');
    btn.style.cssText = `
      width:100%;padding:11px 16px;border-radius:10px;border:none;cursor:pointer;
      background:linear-gradient(135deg,#7c5cff,#00d4ff);color:#071021;
      font-weight:700;font-size:0.9rem;
      transition:opacity .2s ease,transform .2s ease;`;
    btn.innerHTML = '✨ Make My Title Unique (AI Rewrite)';

    btn.onclick = async () => {
      btn.disabled      = true;
      btn.innerHTML     = '<span style="opacity:.7">⏳ AI is rewriting...</span>';
      btn.style.opacity = '0.7';

      try {
        const resp = await fetch('/api/ai/rewrite', {
          method:  'POST',
          headers: { 'Content-Type': 'application/json' },
          body:    JSON.stringify({
            user_title:    originalTitle,
            closest_match: closestMatch || '',
            language:      getSelectedLanguage(),
          }),
        });

        const data = await resp.json();

        if (data.success && data.rewritten_title) {
          showRewriteResult(data.rewritten_title, wrapper);
        } else {
          throw new Error(data.error || 'Rewrite failed');
        }

      } catch (err) {
        btn.innerHTML     = '⚠️ Rewrite failed — try again';
        btn.disabled      = false;
        btn.style.opacity = '1';
        console.error('AI rewrite error:', err);
      }
    };

    wrapper.appendChild(btn);
    aiTitlesList.appendChild(wrapper);
  }

  function showRewriteResult(rewrittenTitle, container) {
    container.innerHTML = `
      <div style="background:linear-gradient(135deg,rgba(124,92,255,0.15),rgba(0,212,255,0.1));
                  border:1px solid rgba(124,92,255,0.4);border-radius:12px;padding:14px;">
        <div style="font-size:11px;font-weight:700;color:#7c5cff;letter-spacing:.06em;
                    margin-bottom:8px;">✨ AI REWRITTEN TITLE</div>
        <div style="font-size:1.05rem;font-weight:700;color:#e6eef8;margin-bottom:12px;">
          ${esc(rewrittenTitle)}
        </div>
        <div style="display:flex;gap:8px;">
          <button onclick="useRewrittenTitle('${esc(rewrittenTitle)}')"
            style="flex:1;padding:9px;border-radius:8px;border:none;cursor:pointer;
                   background:linear-gradient(135deg,#7c5cff,#00d4ff);
                   color:#071021;font-weight:700;font-size:0.85rem;">
            ✅ Use This Title
          </button>
        </div>
      </div>`;
  }

  window.useRewrittenTitle = function (title) {
    if (!titleInput) return;
    titleInput.value   = title;
    lastAnalyzedTitle  = null;
    lastGeneratedTitle = null;
    titleInput.dispatchEvent(new Event('input'));
  };

  // ── Suggestions skeleton ──────────────────────────────────────────────────
  function showSuggestionsSkeleton() {
    if (!aiTitlesList) return;
    aiTitlesList.innerHTML = Array(5).fill(0).map((_, i) => `
      <div class="ai-title-card" style="opacity:.4;">
        <div class="ai-title" style="width:${140 + i * 20}px;height:14px;
             background:linear-gradient(90deg,#1f2937 25%,#2d3748 50%,#1f2937 75%);
             background-size:200% 100%;animation:shimmer 1.4s infinite;border-radius:6px;"></div>
        <div class="use-btn" style="opacity:.3;pointer-events:none;">Use</div>
      </div>`).join('') + `<style>
        @keyframes shimmer{0%{background-position:200% 0}100%{background-position:-200% 0}}
      </style>`;
  }

  // ── Signal row helpers ────────────────────────────────────────────────────
  function signalRow(label, pct, value) {
    return `<tr>
      <td style="padding:4px 8px;color:#94a3b8;">${label}</td>
      <td style="padding:4px 8px;">${miniBar(pct)}</td>
      <td style="padding:4px 8px;color:#e5e7eb;">${value}</td>
    </tr>`;
  }

  function miniBar(pct) {
    const c   = Math.min(100, Math.max(0, pct));
    const col = c > 65 ? '#ef4444' : c > 35 ? '#f59e0b' : '#22c55e';
    return `<div style="width:80px;height:6px;background:#1f2937;border-radius:4px;overflow:hidden;">
              <div style="width:${c}%;height:100%;background:${col};border-radius:4px;"></div>
            </div>`;
  }

  // ── Email Validation ──────────────────────────────────────────────────────
  function handleEmailInput(e) {
    const v  = (e.target.value || '').trim();
    const ok = /^[^@\s]+@[^@\s]+\.[^@\s]+$/.test(v);
    if (emailError) emailError.style.display = (!ok && v.length > 5) ? 'block' : 'none';
    return ok;
  }

  // ── Registration Check ────────────────────────────────────────────────────
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
      if (regError) {
        regError.style.display = 'block';
        regError.textContent   = 'Could not verify registration number.';
      }
    }
  }

  // ── Form Submit ───────────────────────────────────────────────────────────
  async function handleFormSubmit(e) {
    e.preventDefault();

    if (ownerEmail && !handleEmailInput({ target: ownerEmail })) {
      if (formError) {
        formError.style.display = 'block';
        formError.textContent   = 'Please correct the email address before submitting.';
      }
      return;
    }

    const currentTitle = titleInput.value.trim();
    if (!currentTitle) return;

    if (submitBtn) { submitBtn.textContent = 'Submitting…'; submitBtn.disabled = true; }
    showLoading();

    try {
      const analysis = await analyzeTitle(currentTitle);
      renderNLPResults(analysis);

      if (analysis.risk === 'High') {
        hideLoading();
        if (formError) {
          formError.style.display = 'block';
          formError.textContent   = '❌ This title is HIGH RISK and cannot be submitted. Please modify the title.';
        }
        if (submitBtn) { submitBtn.textContent = '🚀 Submit Application'; submitBtn.disabled = false; }
        return;
      }

      if (formError) formError.style.display = 'none';
      submitForm.submit();

    } catch (err) {
      console.error('Submit error:', err);
      hideLoading();
      if (formError) {
        formError.style.display = 'block';
        formError.textContent   = 'Unable to validate title. Please try again.';
      }
      if (submitBtn) { submitBtn.textContent = '🚀 Submit Application'; submitBtn.disabled = false; }
    }
  }

  // ── UI Helpers ────────────────────────────────────────────────────────────
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
    if (analysisSection)      analysisSection.style.display     = 'none';
    if (spinnerContainer)     spinnerContainer.style.display     = 'none';
    if (spinnerContainerArea) spinnerContainerArea.style.display = 'none';
    if (titleSpinner)         titleSpinner.style.display         = 'none';
    if (formOverlay)          formOverlay.style.display          = 'none';

    const empty = document.getElementById('emptyState');
    if (empty) {
      empty.textContent  = 'Start typing to get real-time AI title analysis';
      empty.style.display = 'block';
    }
    lastSimilarity = null;
    lastRisk       = null;
  }

  function getSelectedLanguage() {
    return document.getElementById('languageSelect')?.value || 'English';
  }

  function esc(text) {
    const d = document.createElement('div');
    d.textContent = String(text);
    return d.innerHTML;
  }

  // ── Boot ──────────────────────────────────────────────────────────────────
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();