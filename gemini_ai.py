"""
PRGI AI Module v3.0 — Best Quality Edition
============================================
Primary:  Groq (llama-3.3-70b-versatile)
Fallback: Gemini (gemini-2.5-flash)
Final:    Smart static fallback

KEY IMPROVEMENT in v3.0:
  Two-step suggestion pipeline:
    Step 1 — AI understands the title (language, meaning, domain, script)
    Step 2 — AI generates suggestions anchored to that understanding

  This means:
    "udayavani"        → understands Kannada, "rising voice" → Udaya Vani, Nava Udaya...
    "dainik jagran"    → understands Hindi daily newspaper → Pratap Jagran, Nav Dainik...
    "tech today"       → understands English tech publication → Digital Today, Tech Herald...
    "al balagh"        → understands Urdu/Arabic → relevant Urdu suggestions
    "financial express"→ understands finance domain → Commerce Herald, Arthik Times...

  Works for ALL languages, ALL domains, ALL scripts.
"""

import os
import json
import logging
import urllib.request
import urllib.error
import re

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
GROQ_API_KEY   = os.environ.get("GROQ_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

GROQ_MODEL   = "llama-3.3-70b-versatile"
GEMINI_MODEL = "gemini-2.5-flash"

GROQ_URL   = "https://api.groq.com/openai/v1/chat/completions"
GEMINI_URL = (
    f"https://generativelanguage.googleapis.com/v1beta/models/"
    f"{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
)

MAX_OUTPUT_TOKENS = 512


# ─────────────────────────────────────────────────────────────────────────────
# UTILITY
# ─────────────────────────────────────────────────────────────────────────────
def _clean_title(text: str) -> str:
    text = text.strip()
    text = re.sub(r'^[\s\u2022\-\*0-9)\.]+\s*', '', text)
    text = re.sub(r'[\u2022\-\*:;,\.]+$', '', text)
    return text.strip()


def _validate_title(text: str) -> bool:
    text = _clean_title(text)
    if not text or len(text) > 80 or len(text) < 3:
        return False
    # Allow 1-5 words
    word_count = len(text.split())
    return 1 <= word_count <= 5


def _is_complete_sentence(text: str) -> bool:
    text = text.strip()
    return len(text) >= 40 and len(text.split()) >= 8


def _strip_markdown_json(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = "\n".join(l for l in raw.splitlines() if not l.startswith("```"))
    start = raw.find("{")
    end   = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        return raw[start:end + 1]
    return raw.strip()


# ─────────────────────────────────────────────────────────────────────────────
# CORE: Call Groq
# ─────────────────────────────────────────────────────────────────────────────
def _call_groq(prompt: str, max_tokens: int = MAX_OUTPUT_TOKENS) -> str:
    if not GROQ_API_KEY or not GROQ_API_KEY.strip():
        raise RuntimeError("GROQ_API_KEY not configured.")

    payload = json.dumps({
        "model":       GROQ_MODEL,
        "messages":    [{"role": "user", "content": prompt}],
        "max_tokens":  max_tokens,
        "temperature": 0.8,
        "top_p":       0.95,
    }).encode("utf-8")

    req = urllib.request.Request(
        GROQ_URL,
        data=payload,
        headers={
            "Content-Type":  "application/json",
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "User-Agent":    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                             "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        },
        method="POST"
    )

    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        choices = data.get("choices", [])
        if not choices:
            raise RuntimeError("Groq returned no choices.")
        text = choices[0].get("message", {}).get("content", "").strip()
        if not text or len(text) < 3:
            raise RuntimeError("Groq response empty.")
        log.info(f"Groq success ({len(text)} chars)")
        return text
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        log.error(f"Groq HTTP {e.code}: {body[:300]}")
        raise RuntimeError(f"Groq API error {e.code}: {body[:150]}")
    except urllib.error.URLError as e:
        log.error(f"Groq URL error: {e.reason}")
        raise RuntimeError(f"Groq network error: {e.reason}")
    except json.JSONDecodeError as e:
        log.error(f"Groq JSON decode error: {e}")
        raise RuntimeError("Groq returned invalid JSON.")


# ─────────────────────────────────────────────────────────────────────────────
# CORE: Call Gemini (fallback)
# ─────────────────────────────────────────────────────────────────────────────
def _call_gemini(prompt: str, max_tokens: int = MAX_OUTPUT_TOKENS) -> str:
    if not GEMINI_API_KEY or GEMINI_API_KEY == "YOUR_GEMINI_API_KEY_HERE":
        raise RuntimeError("GEMINI_API_KEY not configured.")

    payload = json.dumps({
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "maxOutputTokens": max_tokens,
            "temperature":     0.8,
            "topP":            0.95,
        },
        "safetySettings": [
            {"category": "HARM_CATEGORY_HARASSMENT",        "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH",       "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]
    }).encode("utf-8")

    req = urllib.request.Request(
        GEMINI_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    try:
        with urllib.request.urlopen(req, timeout=12) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        candidates = data.get("candidates", [])
        if not candidates:
            raise RuntimeError("Gemini returned no candidates.")
        parts = candidates[0].get("content", {}).get("parts", [])
        if not parts:
            raise RuntimeError("Gemini returned empty content.")
        text = parts[0].get("text", "").strip()
        if not text or len(text) < 3:
            raise RuntimeError("Gemini response too short.")
        log.info(f"Gemini fallback success ({len(text)} chars)")
        return text
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        log.error(f"Gemini HTTP {e.code}: {body[:300]}")
        raise RuntimeError(f"Gemini API error {e.code}")
    except urllib.error.URLError as e:
        log.error(f"Gemini URL error: {e.reason}")
        raise RuntimeError(f"Gemini network error: {e.reason}")
    except json.JSONDecodeError:
        raise RuntimeError("Gemini returned invalid JSON.")


# ─────────────────────────────────────────────────────────────────────────────
# UNIFIED CALLER: Groq → Gemini → RuntimeError
# ─────────────────────────────────────────────────────────────────────────────
def _call_ai(prompt: str, max_tokens: int = MAX_OUTPUT_TOKENS) -> tuple:
    """Returns (text, source). Raises RuntimeError if both fail."""
    try:
        return _call_groq(prompt, max_tokens), "groq"
    except RuntimeError as e:
        log.warning(f"Groq failed, trying Gemini. Reason: {e}")
    try:
        return _call_gemini(prompt, max_tokens), "gemini"
    except RuntimeError as e:
        log.warning(f"Gemini also failed. Reason: {e}")
    raise RuntimeError("Both Groq and Gemini failed.")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — UNDERSTAND THE TITLE
# Figures out language, script, meaning, and domain before generating anything.
# This is what makes suggestions relevant for ANY language / ANY domain.
# ─────────────────────────────────────────────────────────────────────────────
def _understand_title(user_title: str) -> dict:
    """
    Analyzes the title to extract:
      - language: e.g. "Kannada", "Hindi", "English", "Urdu", "Tamil"
      - script: e.g. "Latin", "Devanagari", "Kannada script", "Arabic"
      - meaning: plain English meaning of the title
      - domain: publication domain e.g. "general news", "finance", "technology",
                "sports", "politics", "regional", "literature", "science"
      - keywords: 2-3 core thematic words to anchor suggestions

    Returns dict with those keys. Falls back to safe defaults if AI fails.
    """
    prompt = f"""Analyze this Indian publication title: "{user_title}"

Identify exactly:
1. Language it is written in (e.g. Kannada, Hindi, English, Tamil, Telugu, Urdu, Bengali, Marathi, Malayalam, Gujarati, Punjabi, Odia, or Mixed)
2. Script used (Latin, Devanagari, Kannada, Tamil, Telugu, Arabic, Bengali, Gujarati, Gurmukhi, Odia, or Latin-transliteration)
3. Plain English meaning or translation of the title
4. Publication domain: one of [general news, politics, finance, technology, sports, literature, science, regional, agriculture, education, entertainment, health, religion, culture]
5. 2-3 core thematic keywords in English that capture what this publication is about

Respond ONLY in this exact JSON format, nothing else:
{{
  "language": "Kannada",
  "script": "Latin-transliteration",
  "meaning": "Rising voice or voice of dawn",
  "domain": "general news",
  "keywords": ["voice", "dawn", "regional"]
}}

JSON for "{user_title}":"""

    defaults = {
        "language": "Unknown",
        "script":   "Latin",
        "meaning":  user_title,
        "domain":   "general news",
        "keywords": [w for w in user_title.lower().split() if len(w) > 2][:3] or ["news"]
    }

    try:
        raw, _ = _call_ai(prompt, max_tokens=200)
        json_str = _strip_markdown_json(raw)
        data     = json.loads(json_str)

        return {
            "language": str(data.get("language", defaults["language"])),
            "script":   str(data.get("script",   defaults["script"])),
            "meaning":  str(data.get("meaning",  defaults["meaning"])),
            "domain":   str(data.get("domain",   defaults["domain"])),
            "keywords": list(data.get("keywords", defaults["keywords"]))[:3],
        }
    except Exception as e:
        log.warning(f"Title understanding failed: {e}")
        return defaults


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE 1 — AI Title Suggestions (Two-Step Pipeline)
# ─────────────────────────────────────────────────────────────────────────────
def get_ai_title_suggestions(
    user_title: str,
    closest_match: str,
    similarity_percent: float,
    language: str = "English",
    n: int = 5
) -> list:
    """
    Two-step pipeline:
      Step 1: Understand the title (language, meaning, domain, keywords)
      Step 2: Generate n suggestions anchored to that understanding

    Works for any language, any domain, any script.
    Primary: Groq | Fallback: Gemini | Final: Smart static fallback
    """

    # ── Step 1: Understand the title ─────────────────────────────────────────
    understanding = _understand_title(user_title)
    lang     = understanding["language"]
    meaning  = understanding["meaning"]
    domain   = understanding["domain"]
    keywords = understanding["keywords"]
    keywords_str = ", ".join(keywords) if keywords else user_title

    log.info(f"Title understood: lang={lang}, domain={domain}, "
             f"meaning='{meaning}', keywords={keywords}")

    # ── Step 2: Generate suggestions anchored to understanding ───────────────
    conflict_line = (
        f'It conflicts with "{closest_match}" ({round(similarity_percent)}% similar) — '
        f'suggestions must be clearly different from that title too.'
        if closest_match and similarity_percent > 20
        else "Generate fresh creative alternatives."
    )

    # Determine what language to suggest in
    suggest_in = (
        f"in {lang} (or a natural mix of {lang} and English)"
        if lang not in ("Unknown", "English", "Mixed")
        else "in English or a natural English-Indian mix"
    )

    prompt = f"""You are an expert Indian publication naming consultant working for PRGI.

TITLE TO REPLACE: "{user_title}"
WHAT THIS TITLE MEANS: {meaning}
LANGUAGE: {lang}
PUBLICATION DOMAIN: {domain}
CORE THEME KEYWORDS: {keywords_str}
CONFLICT: {conflict_line}

YOUR TASK:
Generate EXACTLY {n} alternative publication titles {suggest_in}.

STRICT REQUIREMENTS:
1. THEME: Every suggestion MUST be about "{domain}" and relate to the meaning "{meaning}"
   — Do NOT suggest generic names unrelated to the theme
2. LANGUAGE: Suggest {suggest_in} — match the style of "{user_title}"
3. DIFFERENT: Each must be clearly different from "{user_title}" and "{closest_match}"
4. LENGTH: 1 to 3 words per title (Indian publications are often short)
5. QUALITY: Sound like a real, credible Indian publication name
6. VARIETY: Each of the {n} titles must have a different structure/prefix
7. NO REPETITION of words across titles where possible

EXAMPLES OF GOOD SUGGESTIONS:
- For "udayavani" (Kannada, general news): Vijaya Vani, Nava Udaya, Prabha Vani, Pratap Vani, Udaya Prakash
- For "dainik jagran" (Hindi, general news): Pratap Dainik, Nav Jagran, Jan Dainik, Rashtriya Pratap, Lok Jagran
- For "financial express" (English, finance): Commerce Herald, Arthik Times, Capital Chronicle, Trade Express, Market Varta
- For "tech today" (English, technology): Digital Herald, Tech Varta, Innovation Times, Cyber Chronicle, Digital Pratap
- For "al balagh" (Urdu, general news): Al Hind, Nav Balagh, Hind Balagh, Al Rashid, Nav Al Hind

Return EXACTLY {n} titles. One per line. No bullets, no numbers, no explanations.

Titles for "{user_title}":"""

    try:
        raw, source = _call_ai(prompt, max_tokens=300)

        lines = [_clean_title(line) for line in raw.splitlines() if line.strip()]
        valid = []
        seen  = {user_title.lower(), (closest_match or "").lower()}

        for title in lines:
            t_low = title.lower()
            # Extra filter: reject if it looks like a repeat of examples we gave
            is_example = any(ex in t_low for ex in [
                "vijaya vani", "nav jagran", "commerce herald",
                "digital herald", "al hind"
            ]) and lang not in ("Kannada", "Hindi", "English", "Urdu")

            if _validate_title(title) and t_low not in seen and not is_example:
                valid.append(title)
                seen.add(t_low)

        # Fill remaining with smart fallback if needed
        if len(valid) < n:
            for fb in _smart_fallback(user_title, understanding):
                if fb.lower() not in seen and len(valid) < n:
                    valid.append(fb)
                    seen.add(fb.lower())

        log.info(f"AI suggestions via {source}: {len(valid)} titles "
                 f"(lang={lang}, domain={domain})")
        return valid[:n]

    except Exception as e:
        log.warning(f"AI title suggestions failed: {e}")
        return _smart_fallback(user_title, understanding)[:n]


def _smart_fallback(user_title: str, understanding: dict = None) -> list:
    """
    Smart static fallback that uses the understood meaning/keywords
    instead of just the raw words.
    """
    if understanding:
        keywords = understanding.get("keywords", [])
        lang     = understanding.get("language", "Unknown")
        domain   = understanding.get("domain", "general news")
    else:
        keywords = [w for w in user_title.split() if len(w) > 2]
        lang     = "Unknown"
        domain   = "general news"

    # Use first meaningful keyword as base
    base = keywords[0].title() if keywords else user_title.split()[0].title()

    # Domain-aware suffixes
    domain_suffixes = {
        "finance":     ["Times", "Herald", "Chronicle", "Arthik", "Capital"],
        "technology":  ["Tech", "Digital", "Innovation", "Cyber", "Times"],
        "sports":      ["Sports", "Krida", "Games", "Champion", "Arena"],
        "education":   ["Vidya", "Shiksha", "Knowledge", "Learning", "Herald"],
        "health":      ["Swastha", "Health", "Arogya", "Wellness", "Herald"],
        "agriculture": ["Krishi", "Kisan", "Farm", "Agri", "Gram"],
        "literature":  ["Sahitya", "Kala", "Culture", "Vani", "Patrika"],
        "religion":    ["Dharma", "Dharm", "Adhyatma", "Spiritual", "Vani"],
    }
    suffixes = domain_suffixes.get(domain, ["Vani", "Herald", "Patrika", "Chronicle", "Times"])

    return [
        f"Nav {base}",
        f"{base} {suffixes[0]}",
        f"Lok {base}",
        f"{base} {suffixes[1]}",
        f"Jan {base} {suffixes[2]}",
    ]


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE 2 — AI Risk Explanation
# ─────────────────────────────────────────────────────────────────────────────
def get_ai_risk_explanation(
    user_title: str,
    closest_match: str,
    similarity_percent: float,
    risk_level: str,
    breakdown: dict
) -> str:
    """
    Natural-language risk explanation specific to this title.
    Primary: Groq | Fallback: Gemini | Final: Static string
    """
    sem      = round(breakdown.get("semantic",  0))
    jaccard  = round(breakdown.get("jaccard",   0))
    phonetic = round(breakdown.get("phonetic",  0))
    edit     = round(breakdown.get("edit",      0))

    active_signals = []
    if sem      > 30: active_signals.append(f"semantic meaning ({sem}%)")
    if jaccard  > 30: active_signals.append(f"word overlap ({jaccard}%)")
    if phonetic > 30: active_signals.append(f"phonetic similarity ({phonetic}%)")
    if edit     > 30: active_signals.append(f"character matching ({edit}%)")
    signals_text = ", ".join(active_signals) if active_signals else "general pattern similarity"

    conflict_line = (
        f'The closest existing PRGI title is "{closest_match}" '
        f'with {round(similarity_percent)}% similarity.'
        if closest_match else
        "No strong conflicting title was found in the PRGI database."
    )

    prompt = f"""You are a senior compliance officer at PRGI (Press Registrar General of India).

Publisher submitted: "{user_title}"
{conflict_line}
Risk level: {risk_level}
Similarity signals: {signals_text}

Write a 2-3 sentence professional explanation FOR THIS SPECIFIC TITLE.

Requirements:
- Mention "{user_title}" by name
- Explain what {risk_level} risk means for registration
- State which signals ({signals_text}) contributed
- Give a clear next action

Style: Plain English, no bullets, no markdown, 60-90 words.

Explanation:"""

    try:
        raw, source = _call_ai(prompt, max_tokens=200)
        if raw and _is_complete_sentence(raw):
            log.info(f"AI risk explanation via {source}")
            return raw.strip()
        raise ValueError(f"Response incomplete ({len(raw)} chars)")
    except Exception as e:
        log.warning(f"AI risk explanation failed: {e}")
        return _fallback_explanation(risk_level, similarity_percent)


def _fallback_explanation(risk_level: str, similarity: float) -> str:
    sim = round(similarity)
    if risk_level == "High":
        return (
            f"Your title has {sim}% similarity with an existing PRGI-registered publication, "
            "which exceeds the acceptable threshold and is likely to result in rejection. "
            "We strongly recommend modifying your title — change the main keywords or add a "
            "unique regional identifier — before resubmitting."
        )
    elif risk_level == "Medium":
        return (
            f"Your title shows {sim}% similarity with existing registered publications, "
            "indicating moderate overlap in meaning or structure. "
            "Consider making your title more distinctive by incorporating unique regional "
            "or thematic elements to strengthen your registration prospects."
        )
    else:
        return (
            "Your title appears sufficiently unique compared to all PRGI-registered publications. "
            "No strong conflicts were detected across semantic, phonetic, or structural analysis. "
            "You may proceed confidently toward registration."
        )


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE 3 — AI Smart Rewrite
# Also uses title understanding for better rewrites
# ─────────────────────────────────────────────────────────────────────────────
def get_ai_smart_rewrite(
    user_title: str,
    closest_match: str,
    language: str = "English"
) -> str:
    """
    Rewrites the title using understanding of its language and meaning.
    Primary: Groq | Fallback: Gemini | Final: Pattern-based string
    """
    # Understand the title first
    understanding = _understand_title(user_title)
    lang    = understanding["language"]
    meaning = understanding["meaning"]
    domain  = understanding["domain"]

    conflict_line = (
        f'It conflicts with "{closest_match}".'
        if closest_match else
        "Make it more distinctive."
    )

    suggest_in = (
        f"in {lang} (matching the style of the original)"
        if lang not in ("Unknown", "Mixed")
        else "in English or a natural Indian language mix"
    )

    prompt = f"""Rewrite this Indian publication title to make it unique.

Original: "{user_title}"
Meaning: {meaning}
Language: {lang}
Domain: {domain}
Problem: {conflict_line}

Create ONE new title {suggest_in} that:
- Keeps the same theme ({domain}) and meaning ({meaning})
- Is clearly different from "{closest_match}"
- Is 1 to 3 words
- Sounds like a real credible Indian publication
- Matches the language style of "{user_title}"

Return ONLY the single rewritten title. Nothing else.

Rewritten title:"""

    try:
        raw, source = _call_ai(prompt, max_tokens=60)
        lines = [_clean_title(line) for line in raw.splitlines() if line.strip()]
        if lines:
            candidate = lines[0]
            if _validate_title(candidate):
                log.info(f"AI rewrite via {source}: '{candidate}'")
                return candidate
        raise ValueError("No valid rewrite in response")
    except Exception as e:
        log.warning(f"AI smart rewrite failed: {e}")
        keywords = understanding.get("keywords", []) if 'understanding' in dir() else []
        base     = keywords[0].title() if keywords else user_title.split()[0].title()
        prefixes = ["Nav", "Jan", "Lok", "Desh", "Bharat", "Pratap"]
        suffixes = ["Chronicle", "Herald", "Express", "Gazette", "Sandesh", "Vani"]
        pfx = prefixes[abs(hash(user_title)) % len(prefixes)]
        sfx = suffixes[abs(hash(user_title + "s")) % len(suffixes)]
        return f"{pfx} {base} {sfx}"


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE 4 — AI Fake Title Detection
# ─────────────────────────────────────────────────────────────────────────────

# Known legitimate single-word Indian publications — never flag these
_KNOWN_PUBLICATIONS = {
    "udayavani", "prajavani", "eenadu", "dinamalar", "dinamani",
    "mathrubhumi", "dainik", "jagran", "bhaskar", "navbharat",
    "lokmat", "sakal", "loksatta", "anandabazar", "pratidin",
    "samaja", "sambad", "deccan", "tribune", "pioneer",
    "statesman", "telegraph", "thehindu", "kesari", "sakal",
    "pudhari", "tarun", "sanmarg", "rashtriya", "navabharat",
}


def get_ai_fake_detection(user_title: str) -> dict:
    """
    AI-powered quality check. Understands Indian language context.
    Whitelist for known publications. Smart heuristic for single-word titles.
    Primary: Groq | Fallback: Gemini | Final: { is_fake: False }
    """
    title_lower = user_title.strip().lower()

    # Whitelist — known real publications
    if title_lower in _KNOWN_PUBLICATIONS:
        log.info(f"Fake detection: '{user_title}' whitelisted as known publication")
        return {"is_fake": False, "reasons": [], "confidence": "Low", "source": "whitelist"}

    # Heuristic for single words with vowels and no caps — likely a real name
    words = user_title.strip().split()
    if len(words) == 1:
        w = words[0]
        has_vowels   = any(c in "aeiouAEIOU" for c in w)
        is_all_caps  = w.isupper() and len(w) > 3
        no_vowels    = not has_vowels and len(w) > 3
        if has_vowels and not is_all_caps and not no_vowels:
            log.info(f"Fake detection: '{user_title}' passed single-word heuristic")
            return {"is_fake": False, "reasons": [], "confidence": "Low", "source": "heuristic"}

    prompt = f"""You are a PRGI quality officer reviewing Indian publication title registrations.

Title to review: "{user_title}"

IMPORTANT CONTEXT:
- Indian publications can be in ANY language: Hindi, Kannada, Tamil, Telugu, Bengali, Marathi, Urdu, Gujarati, Malayalam, Odia, Punjabi, etc.
- Single-word titles in Indian languages are completely legitimate (e.g., Udayavani, Prajavani, Eenadu)
- Transliterated Indian language words that look unusual to English speakers are still legitimate
- Short titles (1-2 words) are normal for Indian publications
- Hindi/regional words like Vani, Patrika, Samachar, Jagran, Bhaskar, Dainik are ALL legitimate

ONLY flag if the title has CLEAR spam/quality issues:
1. Obvious clickbait words: SHOCKING, VIRAL, BREAKING NEWS, MUST READ, EXCLUSIVE ALERT
2. Pure keyboard gibberish: asdfgh, qwerty, zxcvbn patterns
3. Excessive ALL CAPS that is clearly not an acronym or regional word
4. Excessive punctuation: !!!, ???, more than one ! or ?
5. Random numbers with no context: 123abc, xyx123

Do NOT flag:
- Any Indian language words (even if they look unfamiliar)
- Short titles (1-3 words is fine)
- Transliterated regional language titles
- Unusual sounding but plausible publication names

Respond ONLY with valid JSON:
{{"is_suspicious": false, "confidence": "Low", "reasons": []}}

JSON:"""

    try:
        raw, source = _call_ai(prompt, max_tokens=150)
        json_str      = _strip_markdown_json(raw)
        data          = json.loads(json_str)
        is_suspicious = bool(data.get("is_suspicious", False))
        confidence    = str(data.get("confidence", "Low"))
        reasons       = data.get("reasons", [])
        if not isinstance(reasons, list):
            reasons = []
        reasons = [str(r).strip() for r in reasons if r]
        if confidence not in ["Low", "Medium", "High"]:
            confidence = "Low"
        log.info(f"AI fake detection via {source}: suspicious={is_suspicious}")
        return {
            "is_fake":    is_suspicious,
            "reasons":    reasons[:3],
            "confidence": confidence,
            "source":     source,
        }
    except json.JSONDecodeError as e:
        log.warning(f"Fake detection JSON parse error: {e}")
        return {"is_fake": False, "reasons": [], "confidence": "Low", "source": "fallback"}
    except Exception as e:
        log.warning(f"Fake detection failed: {e}")
        return {"is_fake": False, "reasons": [], "confidence": "Low", "source": "fallback"}


# ─────────────────────────────────────────────────────────────────────────────
# SELF-TEST
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  PRGI AI Module v3.0 — Best Quality Self Test")
    print("  Two-step understanding pipeline")
    print("=" * 60)

    test_cases = [
        # (title,               closest_match,            sim,  risk,     breakdown)
        ("udayavani",           "",                        5.0,  "Low",
         {"semantic": 5,  "jaccard": 2,  "phonetic": 3,  "edit": 8}),

        ("dainik jagran",       "dainik bhaskar",         68.0, "Medium",
         {"semantic": 65, "jaccard": 55, "phonetic": 40, "edit": 50}),

        ("financial express",   "financial times",        72.0, "High",
         {"semantic": 70, "jaccard": 60, "phonetic": 35, "edit": 55}),

        ("tech today",          "tech times india",       55.0, "Medium",
         {"semantic": 58, "jaccard": 45, "phonetic": 30, "edit": 40}),

        ("india news today",    "india today news",       82.0, "High",
         {"semantic": 78, "jaccard": 65, "phonetic": 40, "edit": 70}),

        ("krishi patrika",      "krishi jagran",          60.0, "Medium",
         {"semantic": 62, "jaccard": 50, "phonetic": 45, "edit": 48}),

        ("sports chronicle",    "sports times",           55.0, "Medium",
         {"semantic": 55, "jaccard": 40, "phonetic": 30, "edit": 38}),
    ]

    for user_title, match, sim, risk, bd in test_cases:
        print(f"\n{'─'*60}")
        print(f"  Title  : '{user_title}'")
        print(f"  Conflict: '{match}' @ {sim}%  |  Risk: {risk}")
        print(f"{'─'*60}")

        # Show understanding
        und = _understand_title(user_title)
        print(f"  Understood: lang={und['language']} | domain={und['domain']}")
        print(f"              meaning='{und['meaning']}'")
        print(f"              keywords={und['keywords']}")

        print(f"\n  Suggestions:")
        for i, s in enumerate(get_ai_title_suggestions(user_title, match, sim), 1):
            print(f"    {i}. {s}")

        print(f"\n  Explanation:")
        print(f"    {get_ai_risk_explanation(user_title, match, sim, risk, bd)}")

        print(f"\n  Rewrite:")
        print(f"    {get_ai_smart_rewrite(user_title, match)}")

        print(f"\n  Fake Check:")
        r = get_ai_fake_detection(user_title)
        flag = "SUSPICIOUS" if r["is_fake"] else "OK"
        print(f"    [{flag}] source={r['source']} | {r['reasons']}")

    print(f"\n{'─'*60}")
    print("  Fake Detection Edge Cases")
    print(f"{'─'*60}")
    for t in ["BREAKING SHOCKING VIRAL!!!", "Bharat Chronicle",
              "udayavani", "prajavani", "asd qwerty zxc",
              "Eenadu", "Dinamalar", "al balagh", "krishi"]:
        r = get_ai_fake_detection(t)
        print(f"  [{'SUSPICIOUS' if r['is_fake'] else 'OK':10}] '{t}' — {r['source']}")

    print("\n" + "=" * 60)
    print("  Test Complete")
    print("=" * 60)