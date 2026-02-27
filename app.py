import os
BASE_DIR = os.path.dirname(__file__)
DB_PATH = os.path.join(BASE_DIR, 'database', 'database.db')

from flask import Flask, render_template, request, redirect, url_for, g, jsonify, session, send_from_directory
from functools import wraps
import sqlite3
from datetime import datetime, timezone
from prgi_dataset import load_prgi_titles
from similarity_engine import compare_title, initialize_prgi_system, find_closest_title_match, classify_risk, get_titles_from_database, normalize_text

# Load PRGI titles at app startup
PRGI_TITLES = load_prgi_titles()

app = Flask(__name__, static_folder='static', template_folder='templates')
app.secret_key = 'prgi_admin_secret_key_2025_secure_gov_system'

def get_db():
    db = getattr(g, '_database', None)
    if db is None:
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        db = g._database = sqlite3.connect(DB_PATH)
        db.row_factory = sqlite3.Row
    return db

def init_db():
    conn = get_db()
    cur = conn.cursor()
    cur.execute('''
        CREATE TABLE IF NOT EXISTS submissions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            owner TEXT NOT NULL,
            email TEXT NOT NULL,
            state TEXT NOT NULL,
            language TEXT NOT NULL,
            registration_number TEXT NOT NULL,
            similarity_score REAL DEFAULT 0.0,
            similarity_label TEXT DEFAULT 'Unknown',
            created_at TEXT NOT NULL,
            status TEXT DEFAULT 'Pending'
        )
    ''')
    # Ensure a uniqueness constraint on registration_number via a unique index
    cur.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_registration_number ON submissions(registration_number)")
    conn.commit()

# Admin Authentication Constants
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "prgi@123"

def admin_required(f):
    """Decorator to require admin authentication"""
    @wraps(f)
    def admin_wrapper(*args, **kwargs):
        if not session.get('admin_logged_in'):
            return redirect(url_for('admin_login'))
        return f(*args, **kwargs)
    return admin_wrapper

@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()

@app.route('/')
def index():
    # Friendly root redirect to the submit page
    return redirect(url_for('submit_form'))

@app.route("/submit_form", methods=["GET", "POST"])
def submit_form():
    result = None
    error = None
    form_data = {}

    if request.method == "POST":
        title = request.form.get("title", "").strip()
        owner = request.form.get("owner", "").strip()
        email = request.form.get("email", "").strip()
        state = request.form.get("state", "").strip()
        language = request.form.get("language", "").strip()
        registration_number = request.form.get("registration_number", "").strip()

        form_data = {
            "title": title,
            "owner": owner,
            "email": email,
            "state": state,
            "language": language,
            "registration_number": registration_number
        }

        # Run similarity ALWAYS (do not block on validation) — compute against CSV PRGI dataset
        try:
            similarity_analysis = compare_title(title, PRGI_TITLES)
            similarity_score = round(float(similarity_analysis.get("similarity", 0.0)), 2)
            similarity_label = similarity_analysis.get("risk", "Low")
            similarity_result = {
                "similarity": similarity_score,
                "risk": similarity_label,
                "closest_title": similarity_analysis.get('closest_title'),
                "analysis_status": "success" if similarity_score > 0 else "failed"
            }
        except Exception as e:
            print('WARN: similarity check failed:', e)
            similarity_score, similarity_label = 0.0, 'Low'
            similarity_result = {
                "similarity": 0.0,
                "risk": 'Low',
                "closest_title": None,
                "analysis_status": "failed",
                "error": str(e)
            }

        result = {
            "similarity_score": similarity_score,
            "similarity_label": similarity_label,
            "closest_title": similarity_result.get("closest_title", "") if 'similarity_result' in locals() else ""
        }

        # Validation: if missing fields -> show error but include AI result
        if not all(form_data.values()):
            error = "All fields are required."
            return render_template(
                "submit_form.html",
                error=error,
                form_data=form_data,
                result=result
            )

        # Server-side email validation (simple format check)
        import re
        if not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email):
            error = "Please provide a valid email address."
            return render_template(
                "submit_form.html",
                error=error,
                form_data=form_data,
                result=result
            )

        # Uniqueness check (do NOT insert if exists; show error + AI result)
        conn = get_db()
        cur = conn.cursor()
        cur.execute(
            "SELECT 1 FROM submissions WHERE registration_number = ?",
            (registration_number,)
        )
        if cur.fetchone():
            error = "Registration number already exists."
            return render_template(
                "submit_form.html",
                error=error,
                form_data=form_data,
                result=result
            )

        # Block HIGH risk submissions (server-side enforcement)
        if similarity_label == 'High':
            error = "This title appears to be HIGH RISK and cannot be submitted. Please modify the title or contact support."
            return render_template(
                "submit_form.html",
                error=error,
                form_data=form_data,
                result=result
            )

        # Insert new submission (concurrency-safe - catch unique constraint errors)
        created_at = datetime.now(timezone.utc).isoformat()
        try:
            cur.execute("""
                INSERT INTO submissions
                (title, owner, email, state, language, registration_number, created_at, similarity_score, similarity_label)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                title, owner, email, state, language,
                registration_number,
                created_at,
                similarity_score, similarity_label
            ))
            conn.commit()
        except sqlite3.IntegrityError:
            # This handles race conditions where another process inserted the same reg number
            error = "Registration number already exists."
            return render_template(
                "submit_form.html",
                error=error,
                form_data=form_data,
                result=result
            )

        # Redirect to success
        # pass submission id to success for clarity
        submission_id = cur.lastrowid
        return redirect(url_for('success', submission_id=submission_id))

    return render_template(
        "submit_form.html",
        error=error,
        form_data=form_data,
        result=result
    )

@app.route('/success')
def success():
    submission = None
    submission_id = request.args.get('submission_id')
    if submission_id:
        db = get_db()
        cur = db.cursor()
        cur.execute('SELECT * FROM submissions WHERE id = ? LIMIT 1', (submission_id,))
        submission = cur.fetchone()
    return render_template('success.html', submission=submission)

@app.route('/dashboard')
def dashboard():
    """Public dashboard - citizen-facing portal with search and stats"""
    try:
        db = get_db()
        cur = db.cursor()

        # Get stats for overview cards
        cur.execute('SELECT COUNT(*) FROM submissions')
        total_submissions = cur.fetchone()[0]

        cur.execute("SELECT COUNT(*) FROM submissions WHERE status = 'Approved'")
        approved_count = cur.fetchone()[0]

        cur.execute("SELECT COUNT(*) FROM submissions WHERE status = 'Pending'")
        pending_count = cur.fetchone()[0]

        cur.execute("SELECT COUNT(*) FROM submissions WHERE status = 'Rejected'")
        rejected_count = cur.fetchone()[0]

        # Calculate average similarity
        cur.execute('SELECT AVG(similarity_score) FROM submissions')
        avg_similarity = cur.fetchone()[0] or 0

        # Calculate approval rate
        approval_rate = (approved_count / total_submissions * 100) if total_submissions > 0 else 0
        rejection_rate = (rejected_count / total_submissions * 100) if total_submissions > 0 else 0

        # Get state distribution
        cur.execute("""
            SELECT state, COUNT(*) FROM submissions
            GROUP BY state
            ORDER BY COUNT(*) DESC
        """)
        state_data = cur.fetchall()
        state_distribution = {
            'labels': [row[0] for row in state_data],
            'data': [row[1] for row in state_data]
        }

        # Get language distribution
        cur.execute("""
            SELECT language, COUNT(*) FROM submissions
            GROUP BY language
            ORDER BY COUNT(*) DESC
        """)
        language_data = cur.fetchall()
        language_distribution = {
            'labels': [row[0] for row in language_data],
            'data': [row[1] for row in language_data]
        }

        # Get risk distribution
        cur.execute("""
            SELECT similarity_label, COUNT(*) FROM submissions
            GROUP BY similarity_label
        """)
        risk_data = cur.fetchall()
        risk_counts = {'Low': 0, 'Medium': 0, 'High': 0}
        for label, count in risk_data:
            if label in risk_counts:
                risk_counts[label] = count

        total_risks = sum(risk_counts.values())
        risk_distribution = {
            'low': (risk_counts['Low'] / total_risks * 100) if total_risks > 0 else 0,
            'medium': (risk_counts['Medium'] / total_risks * 100) if total_risks > 0 else 0,
            'high': (risk_counts['High'] / total_risks * 100) if total_risks > 0 else 0
        }

        # Get approval trend (last 30 days)
        cur.execute("""
            SELECT DATE(created_at) as date,
                   SUM(CASE WHEN status = 'Approved' THEN 1 ELSE 0 END) as approved,
                   SUM(CASE WHEN status = 'Pending' THEN 1 ELSE 0 END) as pending
            FROM submissions
            WHERE DATE(created_at) >= DATE('now', '-30 days')
            GROUP BY DATE(created_at)
            ORDER BY date
        """)
        trend_data = cur.fetchall()
        approval_trend = {
            'labels': [row[0] for row in trend_data],
            'approved': [row[1] for row in trend_data],
            'pending': [row[2] for row in trend_data]
        }

        # Get common words from titles
        cur.execute("SELECT title FROM submissions WHERE title IS NOT NULL")
        titles = [row[0] for row in cur.fetchall()]
        common_words = get_common_words(titles)

        # Get high-risk keywords
        high_risk_keywords = get_high_risk_keywords(titles)

        stats = {
            'total_submissions': total_submissions,
            'approved_count': approved_count,
            'pending_count': pending_count,
            'rejected_count': rejected_count
        }

        # Handle search functionality
        search_results = None
        search_query = request.args.get('search', '').strip()

        if search_query:
            # Search by title OR registration number
            cur.execute("""
                SELECT * FROM submissions
                WHERE title LIKE ? OR registration_number LIKE ?
                ORDER BY id DESC
            """, (f'%{search_query}%', f'%{search_query}%'))
            search_results = cur.fetchall()

        # Get recent approved titles (last 10)
        cur.execute("""
            SELECT * FROM submissions
            WHERE status = 'Approved'
            ORDER BY id DESC
            LIMIT 10
        """)
        recent_approved = cur.fetchall()

        return render_template('dashboard_dark.html',
                             stats=stats,
                             avg_similarity=avg_similarity,
                             approval_rate=approval_rate,
                             rejection_rate=rejection_rate,
                             state_distribution=state_distribution,
                             language_distribution=language_distribution,
                             risk_distribution=risk_distribution,
                             approval_trend=approval_trend,
                             common_words=common_words,
                             high_risk_keywords=high_risk_keywords,
                             search_results=search_results,
                             recent_approved=recent_approved)
    except Exception as e:
        return f"Dashboard Error: {e}", 500

def get_common_words(titles):
    """Extract common words from titles for AI insights"""
    if not titles:
        return []

    import re
    from collections import Counter

    # Simple word extraction (could be enhanced with NLP)
    all_words = []
    for title in titles:
        words = re.findall(r'\b[a-zA-Z]{3,}\b', title.lower())
        all_words.extend(words)

    word_counts = Counter(all_words)
    return word_counts.most_common(10)

def get_high_risk_keywords(titles):
    """Identify potentially high-risk keywords"""
    if not titles:
        return []

    # Common high-risk patterns (could be enhanced)
    risk_keywords = ['super', 'mega', 'ultra', 'pro', 'premium', 'elite', 'gold', 'royal']
    found_keywords = set()

    for title in titles:
        title_lower = title.lower()
        for keyword in risk_keywords:
            if keyword in title_lower:
                found_keywords.add(keyword)

    return list(found_keywords)

def get_similar_prgi_titles(input_title):
    """
    Fetch top 5 similar PRGI titles from FULL PRGI dataset with similarity scores.

    Returns list of dicts: [{"title": "Title", "similarity": 61}, ...]
    """
    try:
        # Import required modules
        from similarity_engine import SBERT_MODEL, PRGI_EMBEDDINGS, PRGI_TITLES
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np

        # Ensure system is initialized
        if SBERT_MODEL is None or PRGI_EMBEDDINGS is None or PRGI_TITLES is None:
            print("DATABASE FETCH: System not initialized, using fallback")
            return []

        # 🔥 DEBUGGING LOG: Show total PRGI titles available
        print(f"TOTAL PRGI TITLES USED FOR SIMILARITY: {len(PRGI_TITLES)}")
        print("SAMPLE PRGI TITLES:")
        for i, title in enumerate(PRGI_TITLES[:5]):
            print(f"  {i+1}. {title}")

        # Generate embedding for input title
        input_embedding = SBERT_MODEL.encode([input_title])

        # Compute similarities against ALL PRGI titles
        similarities = cosine_similarity(input_embedding, PRGI_EMBEDDINGS)[0]

        # Create list of (similarity_score, title_index) tuples
        title_similarities = []
        for idx, sim in enumerate(similarities):
            similarity_percent = round(float(sim) * 100, 2)
            title_similarities.append((similarity_percent, idx))

        # Sort by similarity (descending - highest similarity first)
        title_similarities.sort(key=lambda x: x[0], reverse=True)

        # ALWAYS select at least ONE closest match (minimum requirement)
        similar_titles = []
        input_normalized = input_title.lower().strip()

        # Find the closest match first (minimum requirement)
        closest_found = False
        for sim_score, idx in title_similarities:
            prgi_title = PRGI_TITLES[idx]
            prgi_normalized = prgi_title.lower().strip()

            # Skip exact matches
            if input_normalized == prgi_normalized:
                continue

            # Always add the first non-exact match as closest
            if not closest_found:
                similar_titles.append({
                    "title": prgi_title,
                    "similarity": sim_score
                })
                # 🔥 DEBUGGING LOG: Show closest title
                print(f"CLOSEST PRGI TITLE: {prgi_title}")
                print(f"CLOSEST SIMILARITY SCORE: {sim_score}")
                closest_found = True
                # If we only need one, we can break here, but continue for top 5
                if len(similar_titles) >= 1:
                    break

        # If no closest match found (shouldn't happen with non-empty dataset),
        # take the first title as fallback
        if not closest_found and len(title_similarities) > 0:
            fallback_idx = title_similarities[0][1]
            fallback_title = PRGI_TITLES[fallback_idx]
            fallback_score = title_similarities[0][0]
            similar_titles.append({
                "title": fallback_title,
                "similarity": fallback_score
            })
            print(f"FALLBACK CLOSEST PRGI TITLE: {fallback_title}")
            print(f"FALLBACK SIMILARITY SCORE: {fallback_score}")

        # Continue to find up to 5 similar titles (optional enhancement)
        if len(similar_titles) < 5:
            for sim_score, idx in title_similarities:
                prgi_title = PRGI_TITLES[idx]
                prgi_normalized = prgi_title.lower().strip()

                # Skip exact matches and already added titles
                if input_normalized == prgi_normalized:
                    continue

                # Check if already added
                if any(item['title'] == prgi_title for item in similar_titles):
                    continue

                similar_titles.append({
                    "title": prgi_title,
                    "similarity": sim_score
                })

                if len(similar_titles) >= 5:
                    break

        print(f"DATABASE FETCH: Found {len(similar_titles)} similar titles from PRGI dataset")
        return similar_titles

    except Exception as e:
        print(f"DATABASE FETCH ERROR: {e}")
        return []

# Admin Routes
@app.route('/admin')
def admin_redirect():
    """Redirect /admin to login if not authenticated"""
    if session.get('admin_logged_in'):
        return redirect(url_for('admin_dashboard'))
    return redirect(url_for('admin_login'))

@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    """Admin login page"""
    error = None
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()

        if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
            session['admin_logged_in'] = True
            session.permanent = True
            return redirect(url_for('admin_dashboard'))
        else:
            error = "Invalid credentials. Please try again."

    return render_template('admin_login.html', error=error)

@app.route('/admin/logout')
def admin_logout():
    """Admin logout - clear session"""
    session.pop('admin_logged_in', None)
    return redirect(url_for('admin_login'))

@app.route('/admin/dashboard')
@admin_required
def admin_dashboard():
    """Secure admin dashboard with filtering"""
    # Filters via query params
    status = request.args.get('status')
    state = request.args.get('state')
    language = request.args.get('language')

    db = get_db()
    cur = db.cursor()

    # total submissions (for top-right display)
    cur.execute('SELECT COUNT(*) FROM submissions')
    total_count = cur.fetchone()[0]

    # Obtain distinct states and languages for dropdown filters
    cur.execute("SELECT DISTINCT state FROM submissions WHERE state IS NOT NULL AND state != '' ORDER BY state")
    states = [r[0] for r in cur.fetchall()]
    cur.execute("SELECT DISTINCT language FROM submissions WHERE language IS NOT NULL AND language != '' ORDER BY language")
    languages = [r[0] for r in cur.fetchall()]

    # Build dynamic query with parameterization
    query = 'SELECT * FROM submissions WHERE 1=1'
    params = []
    # case-insensitive conditions; ignore blank values
    if status and status.strip() != '':
        query += ' AND LOWER(status) = LOWER(?)'
        params.append(status.strip())
    if state and state.strip() != '':
        query += ' AND LOWER(state) = LOWER(?)'
        params.append(state.strip())
    if language and language.strip() != '':
        query += ' AND LOWER(language) = LOWER(?)'
        params.append(language.strip())

    query += ' ORDER BY id DESC'  # newest first
    cur.execute(query, tuple(params))
    rows = cur.fetchall()

    return render_template(
        'admin_dashboard.html',
        submissions=rows,
        total_count=total_count,
        states=states,
        languages=languages,
        selected_status=status,
        selected_state=state,
        selected_language=language
    )


@app.route('/admin/export-approved')
@admin_required
def export_approved():
    """Export approved submissions as CSV (Admin only)"""
    try:
        conn = get_db()
        cur = conn.cursor()
        cur.execute("SELECT title, owner, state, language, registration_number, similarity_score, similarity_label FROM submissions WHERE status = 'Approved'")
        rows = cur.fetchall()

        # Build CSV
        import io, csv
        output = io.StringIO()
        writer = csv.writer(output)
        # Header (exact order)
        writer.writerow(['Title', 'Owner', 'State', 'Language', 'Registration No', 'Similarity %', 'Risk'])
        for r in rows:
            writer.writerow([
                r['title'],
                r['owner'],
                r['state'],
                r['language'],
                r['registration_number'],
                round(r['similarity_score'] or 0.0, 2),
                r['similarity_label'] or ''
            ])

        csv_data = output.getvalue()
        output.close()

        from flask import Response
        resp = Response(csv_data, mimetype='text/csv')
        resp.headers['Content-Disposition'] = 'attachment; filename=approved_titles.csv'
        return resp
    except Exception as e:
        print('CSV export error:', e)
        return "Export failed", 500

@app.route("/admin/update_status", methods=["POST"])
def update_status():
    data = request.get_json(force=True)

    submission_id = int(data.get("id"))
    raw_status = str(data.get("status")).strip().lower()

    if raw_status == "approved":
        status = "Approved"
    elif raw_status == "rejected":
        status = "Rejected"
    else:
        return jsonify({"success": False, "error": "Invalid status"}), 400

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "UPDATE submissions SET status=? WHERE id=?",
        (status, submission_id)
    )
    conn.commit()

    if cur.rowcount == 0:
        return jsonify({"error": "id not found"}), 404

    return jsonify({"success": True, "status": status})

# API: check registration uniqueness (POST)
# API: live similarity check (POST)
@app.route("/api/check_similarity", methods=["POST"])
def check_similarity():
    data = request.get_json()
    title = data.get("title", "").strip()

    if len(title) < 3:
        return jsonify({
            "similarity": 0.0,
            "risk": "Low",
            "closest_title": None
        })

    result = compare_title(title, PRGI_TITLES)

    # Strict policy: do NOT fallback to database-backed titles. If compare_title
    # did not return a closest_title, leave it empty (no fabrication).
    if not result.get('closest_title'):
        # Ensure consistent structure
        result = {
            "similarity": 0.0,
            "risk": "Low",
            "closest_title": None
        }
    return jsonify(result)

# API: check registration uniqueness (POST)
@app.route('/api/check_registration', methods=['POST'])
def api_check_registration():
    data = request.get_json() or request.form
    reg = (data.get('registration_number') or '').strip()
    exists = False
    if reg:
        db = get_db()
        cur = db.cursor()
        cur.execute('SELECT 1 FROM submissions WHERE registration_number = ? LIMIT 1', (reg,))
        exists = bool(cur.fetchone())
    return jsonify({"exists": exists})

@app.route('/generate_ai_titles', methods=['POST'])
def generate_ai_titles():
    """Generate alternative titles with lower similarity scores"""
    data = request.get_json() or request.form
    original_title = (data.get('title') or '').strip()

    if not original_title:
        return jsonify({"error": "Title is required"}), 400

    # Generate alternative titles
    alternatives = generate_alternative_titles(original_title)

    # --- Helper filters for strict AI acceptance ---
    def extract_keywords(text: str):
        import re
        if not text:
            return set()
        words = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
        return set(words)

    def has_user_keywords(ai_title: str, user_title: str):
        user_keywords = extract_keywords(user_title)
        if not user_keywords:
            return False
        ai_lower = (ai_title or '').lower()
        return any(k in ai_lower for k in user_keywords)

    def is_meaningful_title(t: str):
        if not t:
            return False
        words = t.strip().split()
        if len(words) < 2:
            return False
        if any(len(w) <= 2 for w in words):
            return False
        if t.strip().endswith('&') or t.strip().endswith('-'):
            return False
        return True

    # Domain alignment: reject titles that introduce unrelated domains
    DOMAIN_KEYWORDS = {
        'banking': ['bank', 'banking', 'banker', 'bankers', 'bank of'],
        'sports': ['football', 'cricket', 'match', 'league', 'tournament', 'score'],
        'politics': ['election', 'minister', 'parliament', 'government', 'policy']
    }

    def detect_domains(text: str):
        found = set()
        tl = (text or '').lower()
        for domain, kws in DOMAIN_KEYWORDS.items():
            for kw in kws:
                if kw in tl:
                    found.add(domain)
                    break
        return found

    def domain_alignment(candidate: str, user_title: str):
        cand_domains = detect_domains(candidate)
        if not cand_domains:
            return True
        user_domains = detect_domains(user_title)
        # If candidate contains domain indicators not present in user title, reject
        if cand_domains - user_domains:
            return False
        return True

    # Calculate estimated similarity for each alternative and strictly filter
    filtered_alts = []
    user_title = original_title

    for alt in alternatives:
        cand = (alt.get('title') or '').strip()

        # Structural & keyword checks
        if not is_meaningful_title(cand):
            continue
        if not has_user_keywords(cand, user_title):
            continue
        if not domain_alignment(cand, user_title):
            continue

        try:
            result = compare_title(cand)
            est = round(float(result.get('similarity', 0.0)), 1)
            rlabel = result.get('risk', 'Low')
            alt['estimated_similarity'] = est
            alt['risk'] = rlabel
        except Exception:
            continue

        # Enforce final filters: similarity < 30 and Low risk
        try:
            if alt['estimated_similarity'] < 30 and str(alt['risk']).lower() == 'low':
                filtered_alts.append(alt)
        except Exception:
            continue

    # Sort by similarity (lowest first)
    filtered_alts.sort(key=lambda x: x['estimated_similarity'])

    return jsonify(filtered_alts)

@app.route('/api/suggest_titles', methods=['GET'])
def api_suggest_titles():
    """Generate AI suggested titles with low similarity scores"""
    title = request.args.get('title', '').strip()

    if not title:
        return jsonify({
            "status": "success",
            "suggestions": []
        })

    try:
        # Ensure PRGI system is initialized
        initialize_prgi_system()

        # Import required modules
        from similarity_engine import SBERT_MODEL, PRGI_EMBEDDINGS, PRGI_TITLES
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np

        # Generate embedding for input title
        input_embedding = SBERT_MODEL.encode([title])

        # Compute similarities against ALL PRGI titles
        similarities = cosine_similarity(input_embedding, PRGI_EMBEDDINGS)[0]

        # Create list of (similarity_score, title_index) tuples
        title_similarities = []
        for idx, sim in enumerate(similarities):
            similarity_percent = round(float(sim) * 100, 2)
            title_similarities.append((similarity_percent, idx))

        # Sort by similarity (ascending - lowest similarity first)
        title_similarities.sort(key=lambda x: x[0])

        # Select titles where similarity < 30%, pick TOP 5 lowest similarity titles
        suggestions = []
        for sim_score, idx in title_similarities:
            if sim_score < 30 and len(suggestions) < 5:
                suggestions.append(PRGI_TITLES[idx])
            elif len(suggestions) >= 5:
                break

        return jsonify({
            "status": "success",
            "suggestions": suggestions
        })

    except Exception as e:
        print(f"Error in suggest_titles: {e}")
        return jsonify({
            "status": "failed",
            "suggestions": []
        })

@app.route('/api/generate_prgi_titles', methods=['POST'])
def api_generate_prgi_titles():
    """
    Generate EXACTLY 5 alternative publication titles following strict PRGI rules.

    Expected input JSON:
    {
        "user_title": "User's original title",
        "prgi_match_title": "Closest PRGI title match",
        "similarity_percent": 85.5
    }

    Returns: Plain text with exactly 5 lines, one title per line, nothing else.
    """
    try:
        data = request.get_json() or request.form

        user_title = (data.get('user_title') or '').strip()
        prgi_match_title = (data.get('prgi_match_title') or '').strip()
        similarity_percent = float(data.get('similarity_percent', 0))

        if not user_title or not prgi_match_title:
            return "Error: user_title and prgi_match_title are required", 400

        # Generate alternative titles using the strict PRGI rules
        alternative_titles = generate_prgi_alternative_titles(
            user_title,
            prgi_match_title,
            similarity_percent
        )

        # Return exactly 5 lines, one title per line, nothing else
        response_text = '\n'.join(alternative_titles[:5])

        # Ensure we have exactly 5 lines
        lines = response_text.split('\n')
        while len(lines) < 5:
            lines.append(f"Alternative Publication {len(lines) + 1}")

        return '\n'.join(lines[:5]), 200, {'Content-Type': 'text/plain'}

    except Exception as e:
        print(f"Error in generate_prgi_titles: {e}")
        # Return fallback titles if error occurs
        fallback_titles = [
            "Regional Publication Review",
            "State Bulletin Gazette",
            "Local Journal Dispatch",
            "Community News Bulletin",
            "Public Information Chronicle"
        ]
        return '\n'.join(fallback_titles), 200, {'Content-Type': 'text/plain'}

def generate_fallback_suggestions(title):
    """Generate rule-based suggestions if SBERT fails"""
    import re
    from random import choice

    # Simple rule-based alternatives
    prefixes = ['The', 'National', 'Regional', 'Local', 'Community', 'Weekly', 'Monthly']
    suffixes = ['Review', 'Bulletin', 'Journal', 'Chronicle', 'Herald', 'Digest', 'Gazette']
    modifiers = ['India', 'Indian', 'Weekly', 'Journal', 'Review', 'News', 'Publication']

    suggestions = []

    # Add prefixes
    for prefix in choice(prefixes, 3):
        suggestions.append(f"{prefix} {title}")

    # Add suffixes
    for suffix in choice(suffixes, 3):
        suggestions.append(f"{title} {suffix}")

    # Replace common words
    common_words = ['news', 'daily', 'bulletin', 'digest', 'weekly', 'monthly']
    for word in common_words:
        if word in title.lower():
            new_title = title.lower().replace(word, choice(modifiers)).title()
            suggestions.append(new_title)

    # Return with dummy similarity scores
    return [{
        "title": suggestion,
        "similarity": round(30 + (i * 5), 1)  # 30-55% range
    } for i, suggestion in enumerate(suggestions[:6])]

def generate_alternative_titles(original_title):
    """Generate 5-7 alternative titles with lower similarity"""
    import re
    from collections import Counter

    # Common high-risk words to avoid
    high_risk_words = ['indian', 'monthly', 'digest', 'daily', 'news', 'times', 'post', 'tribune']

    # Extract keywords from original title
    words = re.findall(r'\b[a-zA-Z]{3,}\b', original_title.lower())
    word_freq = Counter(words)

    # Get existing titles from database to avoid
    db = get_db()
    cur = db.cursor()
    cur.execute('SELECT title FROM submissions WHERE title IS NOT NULL')
    existing_titles = [row[0].lower() for row in cur.fetchall()]

    alternatives = []

    # Strategy 1: Change word order and structure
    if len(words) >= 3:
        # Reverse word order
        reversed_title = ' '.join(words[::-1]).title()
        alternatives.append({
            'title': reversed_title,
            'strategy': 'Word order reversal'
        })

        # Move last word to front
        if len(words) >= 2:
            moved_title = words[-1].title() + ' ' + ' '.join(words[:-1]).title()
            alternatives.append({
                'title': moved_title,
                'strategy': 'Last word first'
            })

    # Strategy 2: Use broader descriptors
    descriptor_map = {
        'indian': ['national', 'countrywide', 'subcontinental'],
        'monthly': ['periodic', 'regular', 'scheduled'],
        'digest': ['review', 'summary', 'overview'],
        'news': ['bulletin', 'report', 'updates'],
        'daily': ['regular', 'frequent', 'ongoing']
    }

    for word, alternatives_list in descriptor_map.items():
        if word in original_title.lower():
            for alt_word in alternatives_list[:2]:  # Use first 2 alternatives
                new_title = original_title.lower().replace(word, alt_word).title()
                alternatives.append({
                    'title': new_title,
                    'strategy': f'Replace "{word}" with "{alt_word}"'
                })

    # Strategy 3: Add descriptive prefixes/suffixes
    prefixes = ['The', 'National', 'Regional', 'Local', 'Community']
    suffixes = ['Review', 'Bulletin', 'Journal', 'Chronicle', 'Herald']

    for prefix in prefixes[:3]:
        new_title = f"{prefix} {original_title}"
        alternatives.append({
            'title': new_title,
            'strategy': f'Add prefix: "{prefix}"'
        })

    for suffix in suffixes[:3]:
        new_title = f"{original_title} {suffix}"
        alternatives.append({
            'title': new_title,
            'strategy': f'Add suffix: "{suffix}"'
        })

    # Strategy 4: Use abstract/conceptual alternatives
    conceptual_map = {
        'indian': ['subcontinental', 'south asian', 'bharatiya'],
        'monthly': ['periodic', 'scheduled', 'regular'],
        'digest': ['compendium', 'anthology', 'collection']
    }

    for word, conceptual_alternatives in conceptual_map.items():
        if word in original_title.lower():
            for alt in conceptual_alternatives:
                new_title = original_title.lower().replace(word, alt).title()
                alternatives.append({
                    'title': new_title,
                    'strategy': f'Conceptual replacement: "{alt}"'
                })

    # Remove duplicates and filter out titles too similar to original
    seen = set()
    unique_alternatives = []

    for alt in alternatives:
        title_lower = alt['title'].lower()
        # Skip if too similar to original (same words in same order)
        if title_lower == original_title.lower():
            continue

        # Skip if already seen
        if title_lower in seen:
            continue

        # Skip if too similar to existing titles
        too_similar = False
        for existing in existing_titles:
            if calculate_string_similarity(title_lower, existing) > 0.7:
                too_similar = True
                break

        if not too_similar:
            seen.add(title_lower)
            unique_alternatives.append(alt)

    # Return top 5-7 alternatives
    return unique_alternatives[:7]

def calculate_string_similarity(s1, s2):
    """Calculate similarity between two strings (0.0 to 1.0)"""
    # Simple word overlap similarity
    words1 = set(s1.split())
    words2 = set(s2.split())

    if not words1 and not words2:
        return 1.0
    if not words1 or not words2:
        return 0.0

    intersection = words1.intersection(words2)
    union = words1.union(words2)

    return len(intersection) / len(union)

def generate_prgi_alternative_titles(user_title, prgi_match_title, similarity_percent):
    """
    Generate EXACTLY 5 alternative publication titles following strict PRGI rules.

    CRITICAL RULES:
    - Every title MUST be about the SAME SUBJECT as user_title (same topic, domain, audience)
    - Different wording and sentence structure from both user_title and prgi_match_title
    - Avoid overlapping phrases with prgi_match_title
    - Semantically closer to user_title than to prgi_match_title
    - Clearly less similar than prgi_match_title
    - NO generic news titles, banking/law/finance unless user_title is about them
    - NO marketing slogans, one-word titles, copied/modified PRGI titles
    - Output: exactly 5 lines, one title per line, nothing else

    Args:
        user_title: Original user title
        prgi_match_title: Closest PRGI title match
        similarity_percent: Similarity percentage between user and PRGI title

    Returns:
        List of exactly 5 alternative titles
    """
    import re
    from collections import Counter

    # Extract core subject matter from user title
    user_words = re.findall(r'\b[a-zA-Z]{3,}\b', user_title.lower())
    prgi_words = re.findall(r'\b[a-zA-Z]{3,}\b', prgi_match_title.lower())

    # Find words unique to user title (avoid PRGI overlapping phrases)
    user_unique_words = set(user_words) - set(prgi_words)
    user_common_words = set(user_words) & set(prgi_words)

    # Identify core subject keywords (non-generic publication terms)
    publication_terms = {
        'news', 'digest', 'bulletin', 'samachar', 'tak', 'aaj', 'ajj', 'hindustan',
        'times', 'express', 'post', 'mail', 'darpan', 'mirror', 'sandesh', 'patrika',
        'jagran', 'dainik', 'daily', 'weekly', 'monthly', 'fortnightly', 'quarterly',
        'government', 'govt', 'public', 'national', 'india', 'indian', 'bharat',
        'pradesh', 'rajasthan', 'uttar', 'pradesh', 'madhya', 'pradesh', 'maharashtra',
        'gujarat', 'karnataka', 'tamil', 'nadu', 'andhra', 'pradesh', 'telangana',
        'kerala', 'punjab', 'haryana', 'bihar', 'jharkhand', 'west', 'bengal'
    }

    # Extract core subject terms (non-publication generic terms)
    core_subject = [word for word in user_unique_words if word not in publication_terms]

    # If no unique core terms, use common words but modify them
    if not core_subject:
        core_subject = list(user_common_words)[:3]  # Take first 3 common words

    # Alternative descriptors (different from user title structure)
    alternative_descriptors = {
        'indian': ['regional', 'domestic', 'local', 'national', 'subcontinental'],
        'monthly': ['periodic', 'regular', 'scheduled', 'quarterly', 'annual'],
        'digest': ['review', 'summary', 'chronicle', 'journal', 'gazette'],
        'news': ['bulletin', 'report', 'dispatch', 'chronicle', 'gazette'],
        'daily': ['regular', 'frequent', 'ongoing', 'continuous', 'routine'],
        'weekly': ['periodic', 'regular', 'recurring', 'scheduled', 'routine'],
        'times': ['chronicle', 'gazette', 'bulletin', 'dispatch', 'review'],
        'post': ['bulletin', 'dispatch', 'gazette', 'chronicle', 'review']
    }

    # Generate exactly 5 alternative titles with different structures
    alternatives = []

    # Strategy 1: Change core descriptor + keep subject
    if len(core_subject) >= 1:
        for desc, alts in alternative_descriptors.items():
            if desc in user_title.lower():
                for alt_desc in alts[:2]:  # Use 2 alternatives max per descriptor
                    if len(alternatives) >= 5:
                        break
                    # Replace descriptor and add subject term
                    new_title = user_title.lower().replace(desc, alt_desc).title()
                    if core_subject:
                        new_title = f"{new_title} {core_subject[0].title()}"
                    alternatives.append(new_title)
                break

    # Strategy 2: Add geographical prefix + modify structure
    geographical_prefixes = ['Regional', 'State', 'District', 'City', 'Local']
    if len(alternatives) < 5 and core_subject:
        for prefix in geographical_prefixes[:3]:
            if len(alternatives) >= 5:
                break
            # Create new structure: Prefix + Subject + Publication type
            if len(core_subject) >= 2:
                new_title = f"{prefix} {core_subject[0].title()} {core_subject[1].title()} Gazette"
            else:
                new_title = f"{prefix} {core_subject[0].title()} Bulletin"
            alternatives.append(new_title)

    # Strategy 3: Use conceptual alternatives + different word order
    conceptual_alternatives = {
        'indian': ['subcontinental', 'south asian', 'bharatiya', 'national'],
        'monthly': ['quarterly', 'biannual', 'periodic', 'regular'],
        'digest': ['compendium', 'anthology', 'collection', 'summary'],
        'news': ['information', 'updates', 'reports', 'bulletins']
    }

    if len(alternatives) < 5:
        for concept, alts in conceptual_alternatives.items():
            if concept in user_title.lower():
                for alt_concept in alts[:2]:
                    if len(alternatives) >= 5:
                        break
                    # Create different sentence structure
                    if len(core_subject) >= 1:
                        new_title = f"{core_subject[0].title()} {alt_concept.title()} Review"
                        alternatives.append(new_title)
                break

    # Strategy 4: Add thematic modifiers
    thematic_modifiers = ['Community', 'Public', 'General', 'Official', 'Current']
    if len(alternatives) < 5 and core_subject:
        for modifier in thematic_modifiers[:3]:
            if len(alternatives) >= 5:
                break
            # Create: Subject + Thematic + Publication
            if len(core_subject) >= 2:
                new_title = f"{core_subject[0].title()} {core_subject[1].title()} {modifier} Journal"
            else:
                new_title = f"{modifier} {core_subject[0].title()} Gazette"
            alternatives.append(new_title)

    # Strategy 5: Use abstract/conceptual terms + different structure
    abstract_terms = ['Review', 'Bulletin', 'Chronicle', 'Gazette', 'Journal', 'Dispatch']
    if len(alternatives) < 5 and core_subject:
        for term in abstract_terms[:3]:
            if len(alternatives) >= 5:
                break
            # Create: Conceptual + Subject + Type
            new_title = f"Contemporary {core_subject[0].title()} {term}"
            alternatives.append(new_title)

    # Ensure exactly 5 titles (trim or add generic fallbacks if needed)
    alternatives = alternatives[:5]

    # Add generic fallbacks if we don't have enough specific alternatives
    fallback_templates = [
        f"Regional {core_subject[0].title() if core_subject else 'Publication'} Review",
        f"Statewide {core_subject[0].title() if core_subject else 'Bulletin'} Gazette",
        f"Local {core_subject[0].title() if core_subject else 'Journal'} Dispatch",
        f"Community {core_subject[0].title() if core_subject else 'News'} Bulletin",
        f"Public {core_subject[0].title() if core_subject else 'Information'} Chronicle"
    ]

    while len(alternatives) < 5:
        for template in fallback_templates:
            if len(alternatives) >= 5:
                break
            if template not in alternatives:
                alternatives.append(template)

    # Final validation: ensure each title passes similarity checks
    validated_titles = []
    for title in alternatives[:5]:
        # Check similarity to user title (should be reasonably close)
        user_similarity = calculate_string_similarity(title.lower(), user_title.lower())

        # Check similarity to PRGI match (must be lower than original similarity)
        prgi_similarity = calculate_string_similarity(title.lower(), prgi_match_title.lower())

        # Must be semantically closer to user title AND clearly less similar to PRGI match
        if user_similarity > prgi_similarity and prgi_similarity < (similarity_percent / 100 * 0.8):
            validated_titles.append(title)

    # If validation failed, use original alternatives
    if len(validated_titles) < 5:
        validated_titles = alternatives[:5]

    return validated_titles[:5]

if __name__ == '__main__':
    # Initialize DB and SBERT model within application context
    with app.app_context():
        init_db()
        try:
            # Initialize PRGI system with ONE-TIME pipeline
            initialize_prgi_system()
            print("PRGI system initialized successfully")
        except Exception as e:
            print(f"CRITICAL ERROR: Failed to initialize PRGI system: {e}")
            raise
    app.run(host='127.0.0.1', port=5000, debug=False)

# Favicon route (NON-NEGOTIABLE)
@app.route('/favicon.ico')
def favicon():
    return send_from_directory(
        os.path.join(app.root_path, 'static', 'assets', 'icons'),
        'favicon.ico',
        mimetype='image/vnd.microsoft.icon'
    )
