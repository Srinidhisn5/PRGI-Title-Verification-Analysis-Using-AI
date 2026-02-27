/**
 * PRGI AI Title Verification System - Clean Two-Column Layout
 * Single render function, single analysis flow, safe DOM access
 */
(function() {
  'use strict';

  // DOM elements - safely get existing elements
  const titleInput = document.getElementById('publication_title');
  const submitBtn = document.getElementById('submitBtn');
  const rightPanel = document.getElementById('aiPanel');
  const analysisLoader = document.getElementById('aiLoader');
  const closestMatchCard = document.getElementById('closestContainer');
  const closestMatchContent = document.getElementById('closestPrgiBlock');
  const suggestionsCard = document.getElementById('suggestedContainer');
  const suggestionsList = document.getElementById('aiSuggestions');
  const analysisSection = document.getElementById('titleAnalysisSection');
  const spinnerContainerArea = document.getElementById('spinnerContainerArea');

  // New UI elements
  const spinnerContainer = document.getElementById('spinnerContainer');
  const titleSpinner = document.getElementById('titleSpinner');
  const titleSpinnerRing = document.getElementById('titleSpinnerRing');
  const titleSimPercent = document.getElementById('titleSimPercent');
  const titleRiskLabel = document.getElementById('titleRiskLabel');
  const aiGenerateBtn = document.getElementById('aiGenerateBtn');
  const aiTitlesList = document.getElementById('aiSuggestions');
  const ownerEmail = document.getElementById('ownerEmail');
  const emailError = document.getElementById('emailError');
  const regNumber = document.getElementById('regNumber');
  const regError = document.getElementById('regError');
  const submitForm = document.getElementById('submitForm');
  const formError = document.getElementById('formError');

  // State
  let debounceTimer = null;
  let lastSimilarity = null;
  let lastRisk = null;

  // Initialize

  function init() {
    if (!titleInput || !rightPanel || !suggestionsList) {
      return;
    }

    // Set up event listeners
    titleInput.addEventListener('input', debounce(handleTitleAnalysis, 450));
    titleInput.addEventListener('paste', () => { clearTimeout(debounceTimer); debounce(handleTitleAnalysis, 450)({ target: titleInput }); });
    // Also trigger analysis when value replaced or changed programmatically
    titleInput.addEventListener('change', debounce(handleTitleAnalysis, 450));

    if (aiGenerateBtn) aiGenerateBtn.addEventListener('click', handleAIGenerate);

    // Email validation
    if (ownerEmail) ownerEmail.addEventListener('input', handleEmailInput);

    // Registration uniqueness check
    if (regNumber) regNumber.addEventListener('change', checkRegistrationUnique);

    // Form submit handling
    if (submitForm) submitForm.addEventListener('submit', handleFormSubmit);

    setupFormValidation();

    // Initial form validation
    validateForm();
  }

  // Handle AI generation button - only when LOW risk and low similarity
  async function handleAIGenerate(e){
    e && e.preventDefault && e.preventDefault();
    const title = titleInput.value.trim();
    if (!title) return;

    if (aiTitlesList) aiTitlesList.innerHTML = '<div class="muted">Generating AI suggestions…</div>';

    try {
      const resp = await fetch('/generate_ai_titles', {
        method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({ title })
      });
      if (!resp.ok) throw new Error('AI generation failed');
      const arr = await resp.json();

      // Filter: only LOW risk and similarity < 30
      const filtered = arr.filter(a => Number(a.estimated_similarity) < 30 && String(a.risk).toLowerCase() === 'low');

      if (!filtered || filtered.length === 0){
        if (aiTitlesList) aiTitlesList.innerHTML = '<div class="muted">No low-risk AI titles found.</div>';
        return;
      }

      // Render AI title cards
      if (aiTitlesList) {
        aiTitlesList.innerHTML = '';
        filtered.forEach(item => {
          const card = document.createElement('div');
          card.className = 'ai-title-card';

          const t = document.createElement('div'); t.className = 'ai-title-text'; t.textContent = item.title;
          const meta = document.createElement('div'); meta.className = 'ai-title-meta'; meta.textContent = `Est. similarity: ${Math.round(Number(item.estimated_similarity||0))}%`;

          // Add a small spinner ring per AI title (color-coded by risk)
          const spinnerWrap = document.createElement('div'); spinnerWrap.className = 'title-spinner';
          const ring = document.createElement('div'); ring.className = 'spinner-ring';
          if (String(item.risk).toLowerCase() === 'low') ring.classList.add('spinner-low');
          else if (String(item.risk).toLowerCase() === 'medium') ring.classList.add('spinner-medium');
          else ring.classList.add('spinner-high');
          const info = document.createElement('div'); info.className = 'spinner-info'; info.textContent = ` ${Math.round(Number(item.estimated_similarity||0))}% ${item.risk}`;
          spinnerWrap.appendChild(ring); spinnerWrap.appendChild(info);

          const useBtn = document.createElement('button'); useBtn.type='button'; useBtn.className='use-btn'; useBtn.textContent='Use';
          useBtn.addEventListener('click', ()=>{ titleInput.value = item.title; analyzeTitle(item.title).then(renderAnalysisResults).catch(()=>{}); });

          card.appendChild(t); card.appendChild(meta); card.appendChild(spinnerWrap); card.appendChild(useBtn);
          aiTitlesList.appendChild(card);
        });
      }

    } catch (err) {
      if (aiTitlesList) aiTitlesList.innerHTML = '<div class="muted">AI generation failed.</div>';
    }
  }

  // Debounce function
  function debounce(func, wait) {
    return function executedFunction(...args) {
      clearTimeout(debounceTimer);
      debounceTimer = setTimeout(() => func.apply(this, args), wait);
    };
  }

  // Form validation setup
  function setupFormValidation() {
    const requiredFields = ['publication_title', 'ownerName', 'ownerEmail', 'stateSelect', 'languageSelect', 'regNumber'];

    requiredFields.forEach(fieldId => {
      const element = document.getElementById(fieldId);
      if (element) {
        element.addEventListener('input', validateForm);
        element.addEventListener('change', validateForm);
      }
    });
  }

  // Validate form and update submit button
  function validateForm() {
    // Note: Analysis should not be blocked by validation — only submission is blocked
    const requiredFields = ['publication_title', 'ownerName', 'ownerEmail', 'stateSelect', 'languageSelect', 'regNumber'];
    let allValid = true;

    requiredFields.forEach(fieldId => {
      const element = document.getElementById(fieldId);
      const value = element ? element.value.trim() : '';
      if (!value) allValid = false;
    });

    if (submitBtn) {
      if (allValid) {
        submitBtn.classList.remove('disabled');
        submitBtn.disabled = false;
        submitBtn.textContent = '✅ Submit Registration';
      } else {
        submitBtn.classList.add('disabled');
        submitBtn.disabled = true;
        submitBtn.textContent = '📝 Complete All Fields';
      }
    }

    return allValid;
  }

  // Debounced input handler (500-700ms requirement: 600ms)
  async function handleTitleAnalysis(e) {
    const title = e.target.value.trim();
    if (title.length < 3) {
      hideAnalysisResults();
      return;
    }

    showLoading();
    try {
      const data = await analyzeTitle(title);
      renderAnalysisResults(data);
      console.log('Analysis completed, rendering UI');
    } catch (err) {
      // Fail gracefully - keep UI stable
      hideAnalysisResults();
    } finally {
      hideLoading();
    }
  }

  // Validate email format on client
  function handleEmailInput(e){
    const v = (e.target.value || '').trim();
    const ok = /^[^@\s]+@[^@\s]+\.[^@\s]+$/.test(v);
    if (!ok && v.length > 0){
      if (emailError) { emailError.style.display = 'block'; emailError.textContent = 'Please enter a valid email address.'; }
      return false;
    }
    if (emailError) { emailError.style.display = 'none'; emailError.textContent = ''; }
    return true;
  }

  // Check registration uniqueness via existing API
  async function checkRegistrationUnique(e){
    const v = (e.target.value || '').trim();
    if (!v) { if (regError){ regError.style.display = 'none'; regError.textContent=''; } return; }
    try {
      const resp = await fetch('/api/check_registration', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ registration_number: v }) });
      if (!resp.ok) throw new Error('check failed');
      const j = await resp.json();
      if (j.exists){ if (regError){ regError.style.display='block'; regError.textContent='Registration number already exists.'; }} else { if (regError){ regError.style.display='none'; regError.textContent=''; }}
    } catch (err){ if (regError){ regError.style.display='block'; regError.textContent='Could not verify registration uniqueness.'; } }
  }

  // Handle form submit with risk-based control
  async function handleFormSubmit(e){
    e.preventDefault();

    // Basic client validation
    const okEmail = handleEmailInput({ target: ownerEmail });
    if (!okEmail){ if (formError) { formError.style.display='block'; formError.textContent='Please correct the highlighted errors before submitting.'; } return; }

    // Re-run analysis to ensure freshest risk
    const currentTitle = titleInput.value.trim();
    if (!currentTitle) return;

    showLoading();
    try {
      const analysis = await analyzeTitle(currentTitle);
      renderAnalysisResults(analysis);

      // Block submission if HIGH risk
      if (analysis.risk === 'High' || (typeof lastRisk !== 'undefined' && lastRisk === 'High')){
        if (formError){ formError.style.display='block'; formError.textContent='This title appears to be HIGH RISK and cannot be submitted. Please modify the title or contact support.'; }
        return;
      }

      // No blocking: proceed with default submit
      if (formError){ formError.style.display='none'; formError.textContent=''; }
      // Submit the form normally (unobtrusive)
      submitForm.submit();

    } catch (err){
      if (formError){ formError.style.display='block'; formError.textContent='Unable to validate title at this time. Please try again.'; }
    } finally {
      hideLoading();
    }
  }


  // Compute risk according to 0-24 LOW, 25-59 MEDIUM, 60+ HIGH
  function computeRisk(simPercent){
    const p = Number(simPercent) || 0;
    if (p >= 60) return 'High';
    if (p >= 25) return 'Medium';
    return 'Low';
  }

  // Update the spinner under the title with percentage and color-coded ring
  function updateTitleSpinner(simPercent, risk){
    if (!spinnerContainer || !titleSpinner || !titleSpinnerRing || !titleSimPercent || !titleRiskLabel) return;

    // Ensure the spinner container and its area are visible
    if (spinnerContainerArea) spinnerContainerArea.style.display = 'block';
    spinnerContainer.style.display = 'block';

    titleSpinner.style.display = 'flex';
    titleSimPercent.textContent = `${Math.round(Number(simPercent) || 0)}%`;
    titleRiskLabel.textContent = risk;

    // Remove previous classes
    titleSpinnerRing.classList.remove('spinner-low','spinner-medium','spinner-high');

    if (risk === 'Low') titleSpinnerRing.classList.add('spinner-low');
    else if (risk === 'Medium') titleSpinnerRing.classList.add('spinner-medium');
    else titleSpinnerRing.classList.add('spinner-high');

    // store last state
    lastSimilarity = Number(simPercent) || 0;
    lastRisk = risk;
  }

  // Single analysis flow: similarity -> suggestions (one source)
  async function analyzeTitle(title) {
    const simResp = await fetch('/api/check_similarity', {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ title })
    });
    if (!simResp.ok) throw new Error('Similarity check failed');
    const simJson = await simResp.json();

    // Extract similarity and trust backend risk label
    const simPercent = Number(simJson.similarity) || 0;
    const risk = (simJson && simJson.risk) ? simJson.risk : computeRisk(simPercent);

    // Immediately update title spinner for live feedback using backend risk
    try {
      updateTitleSpinner(simPercent, risk);
    } catch (e) {
      // ignore spinner errors
    }

    let suggestions = [];
    try {
      const q = encodeURIComponent(title);
      const sResp = await fetch('/api/suggest_titles?title=' + q);
      if (sResp.ok) {
        const sJson = await sResp.json();
        if (sJson && Array.isArray(sJson.suggestions)) suggestions = sJson.suggestions.slice(0,5);
      }
    } catch (e) {
      // ignore suggestion errors
    }

    // normalize suggestions
    suggestions = suggestions.map(t => normalizeSingleTitle(t))
                             .filter(t => t && t.toLowerCase() !== title.toLowerCase());

    // Use backend-provided DB-only closest match. Do NOT fallback to AI suggestions or user input.
    const closest = simJson.closest_title || null;

    return { closest_title: closest, suggestions, similarity: simPercent, risk };
  }

  // Normalize single title: remove duplicate words while preserving order (case-insensitive)
  function normalizeSingleTitle(t){
    if (!t || typeof t !== 'string') return '';
    const tokens = t.split(/\s+/).filter(Boolean);
    const seen = new Set();
    const out = [];
    for (const token of tokens){
      const lower = token.toLowerCase();
      if (seen.has(lower)) continue;
      seen.add(lower);
      out.push(token);
    }
    return out.join(' ').trim();
  }

  function normalizeTitles(titles){
    return titles.map(normalizeSingleTitle);
  }

  // Single render function - renders to right panel only
  function renderAnalysisResults(data) {
    // Show right panel
    rightPanel.classList.add('visible');

    // Ensure spinner reflects latest values (redundant but safe)
    if (typeof data.similarity !== 'undefined' && typeof data.risk !== 'undefined') {
      try { updateTitleSpinner(data.similarity, data.risk); } catch(e){}
    }

    // Always reveal the analysis section after analysis
    if (analysisSection) {
      analysisSection.style.display = 'block';
    }
    // Ensure spinner container area is visible when analysis runs
    if (spinnerContainerArea) {
      spinnerContainerArea.style.display = 'block';
    }

    // Render closest match
    if (closestMatchCard && closestMatchContent) {
      closestMatchCard.style.display = 'block';

      if (data.closest_title && data.closest_title !== 'None') {
        closestMatchContent.innerHTML = `<span class="closest-match-badge">${escapeHtml(data.closest_title)}</span>`;
      } else {
        // Never display an empty section — show a safe placeholder that indicates no PRGI title loaded
        closestMatchContent.innerHTML = `<span class="closest-match-badge">(No PRGI titles available)</span>`;
      }
    }

    // AI generate button visibility: only when LOW risk (and similarity < 30 optionally)
    if (aiGenerateBtn) {
      if (data.risk === 'Low' && Number(data.similarity) < 30) {
        aiGenerateBtn.style.display = 'inline-block';
      } else {
        aiGenerateBtn.style.display = 'none';
        if (aiTitlesList) aiTitlesList.innerHTML = '';
      }
    }

    // Render AI suggestions
    if (suggestionsCard && suggestionsList) {
      suggestionsCard.style.display = 'block';
      suggestionsList.innerHTML = '';

      if (data.suggestions && data.suggestions.length > 0) {
        data.suggestions.forEach(title => {
          const item = document.createElement('div');
          item.className = 'suggestion-item';

          const textSpan = document.createElement('span');
          textSpan.className = 'suggestion-text';
          textSpan.textContent = title;

          const useBtn = document.createElement('button');
          useBtn.className = 'use-btn';
          useBtn.textContent = 'Use';
          useBtn.type = 'button';
          useBtn.addEventListener('click', () => useSuggestedTitle(title));

          item.appendChild(textSpan);
          item.appendChild(useBtn);
          suggestionsList.appendChild(item);
        });
      } else {
        // Hide suggestions card if none
        suggestionsCard.style.display = 'none';
      }
    }
  }

  // Handle suggested title selection
  function useSuggestedTitle(title) {
    if (titleInput) {
      titleInput.value = title;
      // Trigger analysis again
      titleInput.dispatchEvent(new Event('input', { bubbles: true }));
    }
  }

  // Show loading state
  function showLoading() {
    if (analysisLoader) {
      analysisLoader.style.display = 'block';
    }
  }

  // Hide loading state
  function hideLoading() {
    if (analysisLoader) {
      analysisLoader.style.display = 'none';
    }
  }

  // Hide analysis results (and hide spinner in left panel)
  function hideAnalysisResults() {
    rightPanel.classList.remove('visible');
    if (analysisSection) analysisSection.style.display = 'none';
    if (spinnerContainer) spinnerContainer.style.display = 'none';
    if (spinnerContainerArea) spinnerContainerArea.style.display = 'none';
    if (titleSpinner) {
      titleSpinner.style.display = 'none';
    }
  }

  // HTML escape helper
  function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
  }

  // Initialize when DOM is ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

})();
