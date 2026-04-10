/**
 * PRGI AI Title Verification System - Clean Two-Column Layout
 * Single render function, single analysis flow, safe DOM access
 * ✅ ALL FIXES APPLIED
 */
(function () {
  'use strict';

  // DOM elements - safely get existing elements
  const titleInput = document.getElementById('publication_title');
  const submitBtn = document.getElementById('submitBtn');
  const rightPanel = document.getElementById('aiPanel');
  const analysisLoader = document.getElementById('aiLoader');
  const closestMatchCard = document.getElementById('closestContainer');
  const aiTitlesList = document.getElementById('aiSuggestions');  // ✅ FIX 2: Changed from 'aiTitlesList' to 'aiSuggestions'
  const closestMatchContent = document.getElementById('closestPrgiBlock');
  const suggestionsCard = document.getElementById('suggestedContainer');
  const analysisSection = document.getElementById('titleAnalysisSection');
  const spinnerContainerArea = document.getElementById('spinnerContainerArea');

  // New UI elements
  const spinnerContainer = document.getElementById('spinnerContainer');
  const titleSpinner = document.getElementById('titleSpinner');
  const titleSpinnerRing = document.getElementById('titleSpinnerRing');
  const titleSimPercent = document.getElementById('titleSimPercent');
  const titleRiskLabel = document.getElementById('titleRiskLabel');
  const ownerEmail = document.getElementById('ownerEmail');
  const emailError = document.getElementById('emailError');
  const regNumber = document.getElementById('regNumber');
  const regError = document.getElementById('regError');
  const submitForm = document.getElementById('submitForm');
  const formError = document.getElementById('formError');
  const formOverlay = document.getElementById('formOverlay');  // ✅ FIX: Add form overlay reference

  // State
  let debounceTimer = null;
  let lastSimilarity = null;
  let lastRisk = null;
  let lastAnalyzedTitle = null;  // Added for flickering prevention
  let lastGeneratedTitle = null;
  let currentController = null;
  let isGenerating = false;

  // Initialize

  function init() {
    if (!titleInput || !rightPanel || !aiTitlesList) {
      return;
    }

    // Set up event listeners
    titleInput.addEventListener('input', debounce(handleTitleAnalysis, 450));
    titleInput.addEventListener('paste', () => { clearTimeout(debounceTimer); debounce(handleTitleAnalysis, 450)({ target: titleInput }); });
    // Also trigger analysis when value replaced or changed programmatically
    titleInput.addEventListener('change', debounce(handleTitleAnalysis, 450));


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
  async function handleAIGenerate(e) {
    if (isGenerating) return;
    isGenerating = true;
    e && e.preventDefault && e.preventDefault();
    const title = titleInput.value.trim();
    if (!title) return;

    if (aiTitlesList) aiTitlesList.innerHTML = '<div class="muted">Generating AI suggestions…</div>';

    try {
      const closestMatch = document.getElementById("closestPrgiBlock")?.innerText || "";
      const confidence = lastSimilarity || 0;

      const resp = await fetch('/api/generate_prgi_titles', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          user_title: title,
          prgi_match_title: closestMatch,
          similarity_percent: confidence
        })
      });

      const text = await resp.text();
      const titles = text.split("\n").filter(t => t.trim() !== "");

      if (!titles.length) {
        aiTitlesList.innerHTML = '<div class="muted">No strong AI suggestions found.</div>';
        return;
      }

      // Render BEST AI titles
      aiTitlesList.innerHTML = '';

      titles.forEach(t => {
        const card = document.createElement('div');
        card.className = 'ai-title-card';

        const titleText = document.createElement('div');
        titleText.className = 'ai-title';
        titleText.textContent = t;

        const useBtn = document.createElement('button');
        useBtn.className = 'use-btn';
        useBtn.textContent = 'Use';

        useBtn.onclick = () => {
          titleInput.value = t;
          titleInput.dispatchEvent(new Event('input'));
        };

        card.appendChild(titleText);
        card.appendChild(useBtn);

        aiTitlesList.appendChild(card);
      });

    } catch (err) {
      console.error("Error:", err);
      if (aiTitlesList) aiTitlesList.innerHTML = '<div class="muted">AI generation failed.</div>';
    }  finally {
            isGenerating = false;
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
        submitBtn.textContent = '🚀 Submit Application';
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
    // ✅ IMPROVEMENT: Better flicker prevention - check both title and similarity
    if (title === lastAnalyzedTitle && lastSimilarity !== null) return;
    lastAnalyzedTitle = title;

    // Clear errors on input
    if (formError) formError.style.display = "none";

    showLoading();
    try {
      const data = await analyzeTitle(title);
      renderAnalysisResults(data);
      console.log('Analysis completed, rendering UI');
    } catch (err) {
      console.error("Error:", err);
      // Fail gracefully - keep UI stable
      hideAnalysisResults();
    } finally {
      hideLoading();
    }
  }

  // Validate email format on client
  function handleEmailInput(e) {
    const v = (e.target.value || '').trim();
    const ok = /^[^@\s]+@[^@\s]+\.[^@\s]+$/.test(v);
    if (!ok && v.length > 5) {
      if (emailError) { emailError.style.display = 'block'; emailError.textContent = 'Please enter a valid email address.'; }
      return false;
    }
    if (emailError) { emailError.style.display = 'none'; emailError.textContent = ''; }
    return true;
  }

  // Check registration uniqueness via existing API
  async function checkRegistrationUnique(e) {
    const v = (e.target.value || '').trim();
    if (!v) { if (regError) { regError.style.display = 'none'; regError.textContent = ''; } return; }
    try {
      const resp = await fetch('/api/check_registration', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ registration_number: v }) });
      if (!resp.ok) throw new Error('check failed');
      const j = await resp.json();
      if (j.exists) { if (regError) { regError.style.display = 'block'; regError.textContent = 'Registration number already exists.'; } } else { if (regError) { regError.style.display = 'none'; regError.textContent = ''; } }
    } catch (err) {
      console.error("Error:", err);
      if (regError) { regError.style.display = 'block'; regError.textContent = 'Could not verify registration uniqueness.'; }
    }
  }

  // Handle form submit with risk-based control
  async function handleFormSubmit(e) {
    e.preventDefault();

    // Basic client validation
    const okEmail = handleEmailInput({ target: ownerEmail });
    if (!okEmail) { if (formError) { formError.style.display = 'block'; formError.textContent = 'Please correct the highlighted errors before submitting.'; } return; }

    // Re-run analysis to ensure freshest risk
    const currentTitle = titleInput.value.trim();
    if (!currentTitle) return;

    // Add loading state to button
    if (submitBtn) {
      submitBtn.textContent = "Submitting...";
      submitBtn.disabled = true;
    }

    showLoading();
    try {
      const analysis = await analyzeTitle(currentTitle);
      renderAnalysisResults(analysis);

      // Block submission if HIGH risk
      if (analysis.risk === 'High' || (typeof lastRisk !== 'undefined' && lastRisk === 'High')) {
        hideLoading();  // Hide overlay before showing error
        if (formError) { formError.style.display = 'block'; formError.textContent = 'This title appears to be HIGH RISK and cannot be submitted. Please modify the title or contact support.'; }
        if (submitBtn) {
          submitBtn.textContent = "🚀 Submit Application";
          submitBtn.disabled = false;
        }
        return;
      }

      // No blocking: proceed with default submit
      if (formError) { formError.style.display = 'none'; formError.textContent = ''; }
      // Submit the form normally (unobtrusive)
      submitForm.submit();

    } catch (err) {
      console.error("Error:", err);
      hideLoading();  // Hide overlay before showing error
      if (formError) { formError.style.display = 'block'; formError.textContent = 'Unable to validate title at this time. Please try again.'; }
      if (submitBtn) {
        submitBtn.textContent = "🚀 Submit Application";
        submitBtn.disabled = false;
      }
    }
  }

  // Update the spinner under the title with percentage and color-coded ring
  function updateTitleSpinner(simPercent, risk) {
    if (!spinnerContainer || !titleSpinner || !titleSpinnerRing || !titleSimPercent || !titleRiskLabel) return;

    // Ensure the spinner container and its area are visible
    if (spinnerContainerArea) spinnerContainerArea.style.display = 'block';
    spinnerContainer.style.display = 'block';

    titleSpinner.style.display = 'flex';
    titleSimPercent.textContent = `${Math.round(Number(simPercent) || 0)}%`;
    if (risk === "Low") {
      titleRiskLabel.textContent = "Low Risk ✅";
    } else if (risk === "Medium") {
      titleRiskLabel.textContent = "Medium Risk ⚠";
    } else {
      titleRiskLabel.textContent = "High Risk ❌";
    }

    // Remove previous classes
    titleSpinnerRing.classList.remove('spinner-low', 'spinner-medium', 'spinner-high');

    if (risk === 'Low') titleSpinnerRing.classList.add('spinner-low');
    else if (risk === 'Medium') titleSpinnerRing.classList.add('spinner-medium');
    else titleSpinnerRing.classList.add('spinner-high');

    // store last state
    lastSimilarity = Number(simPercent) || 0;
    lastRisk = risk;
  }

  // Single analysis flow: similarity -> suggestions (one source)
  async function analyzeTitle(title) {
    if (currentController) currentController.abort();
      currentController = new AbortController();
    const simResp = await fetch('/api/check_similarity', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ title }),
      signal: currentController.signal
});
    if (!simResp.ok) throw new Error('Similarity check failed');
    const simJson = await simResp.json();

    // Extract similarity and trust backend risk label
    const simPercent = Number(simJson.similarity) || 0;

    // ALWAYS trust backend risk
    const risk = simJson.risk;
    // Immediately update title spinner for live feedback using backend risk
    try {
      updateTitleSpinner(simPercent, risk);
    } catch (e) {
      // ignore spinner errors
    }

    // Removed duplicate AI suggestions fetch - now only on button click

    // Use backend-provided DB-only closest match. Do NOT fallback to AI suggestions or user input.
    const closest = simJson.closest_title || null;

    return {
      closest_title: closest,
      suggestions: [],
      similarity: simPercent,
      risk,

      // ✅ ADD THESE TWO LINES
      top_matches: simJson.top_matches || [],
      confidence: simJson.confidence || simPercent,

      explanation: simJson.explanation || "No risk explanation available.",
      fake_warning: simJson.fake_warning || false,
      fake_reasons: simJson.fake_reasons || []
    };
  }

  // Normalize single title: remove duplicate words while preserving order (case-insensitive)
  function normalizeSingleTitle(t) {
    if (!t || typeof t !== 'string') return '';
    const tokens = t.split(/\s+/).filter(Boolean);
    const seen = new Set();
    const out = [];
    for (const token of tokens) {
      const lower = token.toLowerCase();
      if (seen.has(lower)) continue;
      seen.add(lower);
      out.push(token);
    }
    return out.join(' ').trim();
  }

  // Single render function - renders to right panel only
  function renderAnalysisResults(data) {
    // ✅ FIX 1: SHOW RIGHT PANEL AT TOP
    if (rightPanel) {
      rightPanel.classList.add('visible');   // ✅ SHOW PANEL
    }

    // Ensure spinner reflects latest values (redundant but safe)
    if (typeof data.similarity !== 'undefined' && typeof data.risk !== 'undefined') {
      try { updateTitleSpinner(data.similarity, data.risk); } catch (e) { }
    }

    // Always reveal the analysis section after analysis
    if (analysisSection) {
      analysisSection.style.display = 'block';
      const emptyState = document.getElementById('emptyState');
      if (emptyState) emptyState.style.display = 'none';
    }
    // Ensure spinner container area is visible when analysis runs
    if (spinnerContainerArea) {
      spinnerContainerArea.style.display = 'block';
    }

    // Render closest match
if (closestMatchCard && closestMatchContent) {

  const bestMatch = data.top_matches?.[0]?.[0];
  const bestScore = data.top_matches?.[0]?.[1];

  closestMatchCard.style.display = 'block';

// ❗ ADD THIS CONDITION
  if (!bestMatch || (bestScore || 0) < 10) {
    closestMatchContent.innerHTML = `
      <span class="no-match-text">No strong PRGI match found</span>
    `;
  } else {
    closestMatchContent.innerHTML = `
      <span class="closest-match-badge">${escapeHtml(bestMatch)}</span>
      <div style="margin-top:6px; font-size:13px; color:#94a3b8;">
        Similarity: ${Math.round(bestScore || 0)}%
      </div>
    `;
  }
}
    const riskText = document.getElementById("riskExplanationText");
    const riskBox = document.getElementById("riskExplanationContainer");
    if (riskText && riskBox) {
      if (data.risk) {
        const breakdown = data.top_matches?.[0]?.[2] || {};
        const semantic = breakdown.semantic || 0;
        const keywords = breakdown.jaccard || 0;
        const phonetic = (breakdown.phonetic || 0) > 50 ? "Yes" : "No";
        const structure = (breakdown.structure || 0) > 70 ? "Yes" : "No";

        let simpleMsg = "";

        if (data.risk === "Low") {
          simpleMsg = "✅ This title is unique and safe to use.";
        } else if (data.risk === "Medium") {
          simpleMsg = "⚠ This title is somewhat similar to existing titles.";
        } else {
          simpleMsg = "❌ High risk: very similar to an existing registered title.";
        }

        const cleanPercent = value => {
          const n = Number(value) || 0;
          return n <= 1 ? Math.round(n * 100) : Math.round(n);
        };

        const semanticPct = Math.round(breakdown.semantic || 0);
        const keywordPct = Math.round(breakdown.jaccard || 0);
        const phoneticPct = Math.round(breakdown.phonetic || 0);
        const structurePct = Math.round(breakdown.structure || 0);

       let insight = "";

      if (data.similarity >= 65) {
        insight = "Strong similarity detected with an existing PRGI title.";
      } else if (data.similarity >= 35) {
        insight = "Moderate similarity detected — may require review.";
      } else {
        insight = "No significant similarity with existing titles.";
      }

        let keywordText = keywordPct > 0
          ? `${keywordPct}% similarity in words used`
          : "No significant shared words";

        let phoneticText = phoneticPct > 50
          ? "Sounds very similar to an existing title"
          : phoneticPct > 20
            ? "Slight phonetic resemblance"
            : "No phonetic similarity detected";

        let structureText = structurePct > 70
          ? "Follows similar title structure"
          : "Different title structure";

        let html = `
          <div style="margin-bottom:10px; font-weight:700;">
            ${simpleMsg}
          </div>

          <div style="margin-bottom:12px; color:#cbd5e1;">
            ${insight} (${semanticPct}% semantic similarity)
          </div>

          <div style="font-size:14px; line-height:1.7;">
            <b style="color:#38bdf8;">Confidence Score:</b> ${data.confidence || 0}%<br><br>

            <b style="color:#a78bfa;">AI Analysis</b><br>
            • ${keywordText}<br>
            • ${phoneticText}<br>
            • ${structureText}<br>
          </div>
        `;

        riskText.innerHTML = html;

        riskBox.style.display = "block";
      } else {
        riskBox.style.display = "none";
      }
    }
    const fakeBox = document.getElementById("fakeTitleWarning");
    const fakeText = document.getElementById("fakeTitleReasons");

    if (data.fake_warning && fakeBox && fakeText) {

      fakeText.innerHTML = data.fake_reasons
        .map(r => `• ${r}`)
        .join("<br>");

      fakeBox.style.display = "block";

    } else if (fakeBox) {

      fakeBox.style.display = "none";

    }

    // Show suggestions container
    if (suggestionsCard && aiTitlesList) {

      if (data.risk !== 'High') 

        suggestionsCard.style.display = 'block';

        // ✅ Prevent repeated API calls
        if (titleInput.value !== lastGeneratedTitle) {

          aiTitlesList.innerHTML = '<div class="muted">Generating smart AI titles...</div>';

          lastGeneratedTitle = titleInput.value;

          handleAIGenerate();
        }

      } else {
        suggestionsCard.style.display = 'none';
        aiTitlesList.innerHTML = '';
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

  // ✅ FIX 1: ENHANCED SHOW LOADING - Show panel + loader + form overlay + hide old results
  function showLoading() {
    if (rightPanel) rightPanel.classList.add('visible');  // Show right panel

    if (analysisLoader) {
      analysisLoader.style.display = 'block';
    }

    if (formOverlay) {
      formOverlay.style.display = 'block';  // ✅ FIX: Show form overlay during loading
    }

    if (analysisSection) {
      analysisSection.style.display = 'none';   // Hide old results to prevent stale data
    }
  }

  // Hide loading state
  function hideLoading() {
    if (analysisLoader) {
      analysisLoader.style.display = 'none';
    }

    if (formOverlay) {
      formOverlay.style.display = 'none';  // ✅ FIX: Hide form overlay when loading complete
    }
  }

  // Hide analysis results (and hide spinner in left panel)
  function hideAnalysisResults() {
    if (analysisSection) analysisSection.style.display = 'none';
    if (spinnerContainer) spinnerContainer.style.display = 'none';
    if (spinnerContainerArea) spinnerContainerArea.style.display = 'none';
    if (titleSpinner) {
      titleSpinner.style.display = 'none';
    }
    if (formOverlay) {
      formOverlay.style.display = 'none';  // ✅ FIX: Hide overlay when hiding results
    }
    const emptyState = document.getElementById('emptyState');
    if (emptyState) {
      emptyState.innerHTML = 'Start typing to get real-time AI title analysis';
      emptyState.style.display = 'block';
    }
  }
  // AI typing animation removed

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
  // ✅ CLEANUP: Removed duplicate submit handler that was at the bottom
})();