"""
PRGI AI Routes — api_ai_routes.py
===================================
All Gemini-powered API endpoints.
Register in app.py with: app.register_blueprint(ai_bp)

Endpoints added:
  POST /api/ai/suggestions   — AI title suggestions (replaces /api/generate_prgi_titles)
  POST /api/ai/explanation   — AI risk explanation
  POST /api/ai/rewrite       — AI smart rewrite ("Make My Title Unique")
  POST /api/ai/fake_detect   — AI fake/spam detection
  POST /api/ai/full_analysis — Combined: explanation + fake detect in one call
                               (used by submit.js for real-time analysis)
"""

import logging
from flask import Blueprint, request, jsonify
from flask_wtf.csrf import CSRFProtect

log = logging.getLogger(__name__)

ai_bp = Blueprint("ai", __name__, url_prefix="/api/ai")

# Lazy import so app.py doesn't need to change import order
def _gemini():
    import gemini_ai
    return gemini_ai


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: CSRF exempt all AI routes (called via JS fetch)
# ─────────────────────────────────────────────────────────────────────────────
def exempt_from_csrf(csrf: CSRFProtect, blueprint: Blueprint):
    """Call this from app.py after creating csrf: exempt_from_csrf(csrf, ai_bp)"""
    csrf.exempt(blueprint)


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINT 1 — AI Title Suggestions
# POST /api/ai/suggestions
# Body: { user_title, closest_match, similarity_percent, language (optional) }
# Returns: { success, suggestions: ["Title 1", ...], source: "gemini"|"fallback" }
# ─────────────────────────────────────────────────────────────────────────────
@ai_bp.route("/suggestions", methods=["POST"])
def ai_suggestions():
    data = request.get_json(silent=True) or {}

    user_title   = (data.get("user_title")     or "").strip()
    closest      = (data.get("closest_match")  or "").strip()
    sim_pct      = float(data.get("similarity_percent", 0) or 0)
    language     = (data.get("language")       or "English").strip()

    if not user_title:
        return jsonify({"success": False, "error": "user_title required"}), 400

    try:
        g = _gemini()
        suggestions = g.get_ai_title_suggestions(
            user_title   = user_title,
            closest_match = closest,
            similarity_percent = sim_pct,
            language     = language,
            n            = 5
        )
        return jsonify({
            "success":     True,
            "suggestions": suggestions,
            "source":      "gemini"
        })

    except Exception as e:
        log.error(f"AI suggestions error: {e}")
        return jsonify({
            "success":     False,
            "suggestions": [],
            "error":       str(e)
        }), 500


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINT 2 — AI Risk Explanation
# POST /api/ai/explanation
# Body: { user_title, closest_match, similarity_percent, risk_level, breakdown }
# Returns: { success, explanation: "..." }
# ─────────────────────────────────────────────────────────────────────────────
@ai_bp.route("/explanation", methods=["POST"])
def ai_explanation():
    data = request.get_json(silent=True) or {}

    user_title   = (data.get("user_title")    or "").strip()
    closest      = (data.get("closest_match") or "").strip()
    sim_pct      = float(data.get("similarity_percent", 0) or 0)
    risk_level   = (data.get("risk_level")    or "Low").strip()
    breakdown    = data.get("breakdown")      or {}

    if not user_title:
        return jsonify({"success": False, "error": "user_title required"}), 400

    try:
        g = _gemini()
        explanation = g.get_ai_risk_explanation(
            user_title         = user_title,
            closest_match      = closest,
            similarity_percent = sim_pct,
            risk_level         = risk_level,
            breakdown          = breakdown
        )
        return jsonify({"success": True, "explanation": explanation})

    except Exception as e:
        log.error(f"AI explanation error: {e}")
        return jsonify({"success": False, "explanation": "", "error": str(e)}), 500


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINT 3 — AI Smart Rewrite
# POST /api/ai/rewrite
# Body: { user_title, closest_match, language (optional) }
# Returns: { success, rewritten_title: "..." }
# ─────────────────────────────────────────────────────────────────────────────
@ai_bp.route("/rewrite", methods=["POST"])
def ai_rewrite():
    data = request.get_json(silent=True) or {}

    user_title = (data.get("user_title")    or "").strip()
    closest    = (data.get("closest_match") or "").strip()
    language   = (data.get("language")      or "English").strip()

    if not user_title:
        return jsonify({"success": False, "error": "user_title required"}), 400

    try:
        g = _gemini()
        rewritten = g.get_ai_smart_rewrite(
            user_title    = user_title,
            closest_match = closest,
            language      = language
        )
        return jsonify({"success": True, "rewritten_title": rewritten})

    except Exception as e:
        log.error(f"AI rewrite error: {e}")
        return jsonify({"success": False, "rewritten_title": "", "error": str(e)}), 500


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINT 4 — AI Fake Detection
# POST /api/ai/fake_detect
# Body: { user_title }
# Returns: { success, is_fake, reasons, confidence, source }
# ─────────────────────────────────────────────────────────────────────────────
@ai_bp.route("/fake_detect", methods=["POST"])
def ai_fake_detect():
    data = request.get_json(silent=True) or {}

    user_title = (data.get("user_title") or "").strip()

    if not user_title:
        return jsonify({"success": False, "error": "user_title required"}), 400

    try:
        g = _gemini()
        result = g.get_ai_fake_detection(user_title)
        result["success"] = True
        return jsonify(result)

    except Exception as e:
        log.error(f"AI fake detection error: {e}")
        return jsonify({
            "success":    False,
            "is_fake":    False,
            "reasons":    [],
            "confidence": "Low",
            "source":     "fallback",
            "error":      str(e)
        }), 500


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINT 5 — Full AI Analysis (combines explanation + fake detect)
# POST /api/ai/full_analysis
# Body: { user_title, closest_match, similarity_percent, risk_level, breakdown }
# Returns: { success, explanation, is_fake, fake_reasons, fake_confidence }
#
# This is the main endpoint called by submit.js on every title keystroke.
# Runs explanation + fake detection in sequence (could be parallelized later).
# ─────────────────────────────────────────────────────────────────────────────
@ai_bp.route("/full_analysis", methods=["POST"])
def ai_full_analysis():
    data = request.get_json(silent=True) or {}

    user_title   = (data.get("user_title")    or "").strip()
    closest      = (data.get("closest_match") or "").strip()
    sim_pct      = float(data.get("similarity_percent", 0) or 0)
    risk_level   = (data.get("risk_level")    or "Low").strip()
    breakdown    = data.get("breakdown")      or {}

    if not user_title or len(user_title) < 3:
        return jsonify({
            "success":          False,
            "explanation":      "",
            "is_fake":          False,
            "fake_reasons":     [],
            "fake_confidence":  "Low"
        }), 400

    try:
        g = _gemini()

        # Run both in sequence
        explanation = g.get_ai_risk_explanation(
            user_title         = user_title,
            closest_match      = closest,
            similarity_percent = sim_pct,
            risk_level         = risk_level,
            breakdown          = breakdown
        )

        fake_result = g.get_ai_fake_detection(user_title)

        return jsonify({
            "success":         True,
            "explanation":     explanation,
            "is_fake":         fake_result.get("is_fake",    False),
            "fake_reasons":    fake_result.get("reasons",    []),
            "fake_confidence": fake_result.get("confidence", "Low"),
            "fake_source":     fake_result.get("source",     "fallback"),
        })

    except Exception as e:
        log.error(f"AI full analysis error: {e}")
        return jsonify({
            "success":         False,
            "explanation":     "",
            "is_fake":         False,
            "fake_reasons":    [],
            "fake_confidence": "Low",
            "error":           str(e)
        }), 500