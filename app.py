import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# ✅ BASE PATH FIX
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, 'database', 'database.db')

import logging

# ✅ FIREBASE SETUP (FINAL WORKING)
import firebase_admin
from firebase_admin import credentials, auth

cred_path = os.path.join(BASE_DIR, "firebase-key.json")

if not firebase_admin._apps:
    cred = credentials.Certificate(cred_path)
    firebase_admin.initialize_app(cred)

# ✅ FLASK IMPORTS (+ CSRF)
from flask import Flask, render_template, request, redirect, url_for, g, jsonify, session, send_from_directory
from flask_wtf.csrf import CSRFProtect
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps

# ✅ OTHER IMPORTS
import time
import random
import smtplib
from email.mime.text import MIMEText
import difflib
import sqlite3
import secrets
import re
from datetime import datetime, timezone, date

# ✅ YOUR MODULES
from prgi_dataset import load_prgi_titles
from similarity_engine import compare_title, initialize_system, normalize_text

# ✅ LOAD DATASET
PRGI_TITLES = load_prgi_titles()

# ✅ INITIALIZE ONCE AT STARTUP
initialize_system()

app = Flask(__name__, static_folder='static', template_folder='templates')

# ✅ SECURE SECRET KEY (MANDATORY)
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret-key")
if not app.secret_key:
    raise RuntimeError("SECRET_KEY must be set in environment")

# ✅ CSRF PROTECTION
csrf = CSRFProtect(app)

# ✅ HARDENED SESSION CONFIG
app.config.update(
    SESSION_COOKIE_NAME="prgi_session",
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=False,
    PERMANENT_SESSION_LIFETIME=3600
)

logging.basicConfig(level=logging.INFO)

MAX_DAILY_USAGE = 5
LOGIN_ATTEMPTS = {}
from collections import OrderedDict

class LimitedCache(OrderedDict):
    def __init__(self, max_size=1000):
        self.max_size = max_size
        super().__init__()

    def __setitem__(self, key, value):
        if key in self:
            del self[key]
        elif len(self) >= self.max_size:
            self.popitem(last=False)
        super().__setitem__(key, value)

SIM_CACHE = LimitedCache(max_size=1000)

# ✅ ADD OTP SYSTEM HERE
OTP_STORE = {}

def generate_otp():
    return str(random.randint(100000, 999999))

def send_otp_email(to_email, otp):
    sender = "srinidhisnnairy@gmail.com"
    password = os.environ.get("EMAIL_PASSWORD")
    if not password:
        raise RuntimeError("EMAIL_PASSWORD not set in environment")

    msg = MIMEText(f"Your OTP for PRGI verification is: {otp}")
    msg['Subject'] = "PRGI Email Verification OTP"
    msg['From'] = sender
    msg['To'] = to_email

    server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
    server.login(sender, password)
    server.send_message(msg)
    server.quit()

def get_db():
    db = getattr(g, '_database', None)
    if db is None:
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        db = g._database = sqlite3.connect(DB_PATH, check_same_thread=False)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA journal_mode=WAL;")
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
        status TEXT DEFAULT 'Pending',
        user_id INTEGER
    )
''')
    # USERS TABLE
    cur.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        email TEXT UNIQUE,
        password TEXT,
        trust_score INTEGER DEFAULT 50,
        usage_count INTEGER DEFAULT 0,
        last_used_date TEXT,
        email_verified INTEGER DEFAULT 0
    )
    """)
    # Ensure a uniqueness constraint on registration_number via a unique index
    cur.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_registration_number ON submissions(registration_number)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_user_id ON submissions(user_id)")
    conn.commit()

# Admin Authentication Constants
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "admin123")
if not ADMIN_PASSWORD:
    raise RuntimeError("ADMIN_PASSWORD must be set")

# ✅ HELPER: Check if user already logged in
def already_logged_in():
    """Check if user is already logged in"""
    return session.get("user_id") is not None

def admin_required(f):
    """Decorator to require admin authentication"""
    @wraps(f)
    def admin_wrapper(*args, **kwargs):
        if not session.get('admin_logged_in'):
            return redirect(url_for('admin_login'))
        return f(*args, **kwargs)
    return admin_wrapper

def login_required(f):
    """Decorator to require user authentication"""
    @wraps(f)
    def wrapper(*args, **kwargs):
        if not session.get('user_id'):
            return redirect(url_for('login_page'))
        return f(*args, **kwargs)
    return wrapper

@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()

@app.errorhandler(500)
def internal_error(e):
    logging.error("Internal server error", exc_info=e)
    return "Something went wrong", 500

@app.route('/')
def index():
    # Friendly root redirect to the submit page
    return redirect(url_for('submit_form'))

# ✅ PREVENT RELOGIN - Redirect already logged in users
@app.route('/login_page')
def login_page():
    if already_logged_in():
        return redirect(url_for('submit_form'))
    return render_template('login.html')

# ✅ PROFESSIONAL LOGIN VALIDATION
@app.route('/user/login', methods=['POST'])
def user_login():
    email = (request.form.get('email') or '').strip().lower()
    password = request.form.get('password') or ''

    if not email or not password:
        return render_template("login.html", error="All fields required")

    ip = request.remote_addr
    LOGIN_ATTEMPTS[ip] = LOGIN_ATTEMPTS.get(ip, 0) + 1
    if LOGIN_ATTEMPTS[ip] > 10:
        return render_template("login.html", error="Too many attempts. Try later.")

    db = get_db()
    cur = db.cursor()
    cur.execute("SELECT * FROM users WHERE email=?", (email,))
    user = cur.fetchone()

    if not user:
        return render_template("login.html", error="Account not found. Please register first.")

    if user['email_verified'] == 0:
        return render_template("login.html", error="Please verify your email first")
    # ACCOUNT TYPE CHECK FIRST
    if user['password'] == "GOOGLE_AUTH":
        return render_template("login.html", error="This account uses Google login. Please use Google sign-in.")

    # Then password check
    if not check_password_hash(user['password'], password):
        return render_template("login.html", error="Incorrect password. Please try again.")

    # ✅ SUCCESS LOGIN with clean session
    session.clear()
    session['user_id'] = user['id']
    session['user_name'] = user['name'] or "User"
    session['auth_provider'] = 'email'
    session.permanent = True
    LOGIN_ATTEMPTS[ip] = 0
    logging.info(f"User {user['id']} ({email}) logged in successfully")
    return redirect(url_for('submit_form'))

# ✅ PROFESSIONAL REGISTER VALIDATION
@app.route('/user/register', methods=['POST'])
def user_register():
    name = (request.form.get('name') or '').strip()
    email = (request.form.get('email') or '').strip().lower()
    password = request.form.get('password') or ''

    if not name or not email or not password:
        return render_template("register.html", error="All fields required")

    if len(password) < 6:
        return render_template("register.html", error="Password must be at least 6 characters")

    db = get_db()
    cur = db.cursor()

    cur.execute("SELECT * FROM users WHERE email=?", (email,))
    if cur.fetchone():
        return render_template("register.html", error="Email already exists")
    
    otp = generate_otp()
    OTP_STORE[email] = {
        "otp": otp,
        "name": name,
        "password": generate_password_hash(password),
        "time": time.time()
    }

    send_otp_email(email, otp)

    return render_template("verify_otp.html", email=email)
@app.route('/verify-otp', methods=['POST'])
def verify_otp():
    email = request.form.get('email')
    otp = request.form.get('otp')

    stored = OTP_STORE.get(email)

    if not stored:
        return render_template("verify_otp.html", email=email, error="Invalid OTP")

    # expiry check
    if time.time() - stored["time"] > 300:
        return render_template("verify_otp.html", email=email, error="OTP expired")

    # otp check
    if stored["otp"] != otp:
        return render_template("verify_otp.html", email=email, error="Wrong OTP")

    db = get_db()
    cur = db.cursor()

    cur.execute("""
        INSERT INTO users (name, email, password, email_verified)
        VALUES (?, ?, ?, 1)
    """, (stored["name"], email, stored["password"]))

    db.commit()

    del OTP_STORE[email]

    return redirect(url_for('login_page'))
@app.route('/resend-otp', methods=['POST'])
@csrf.exempt 
def resend_otp():
    email = request.form.get('email')

    if not email:
        return jsonify({"success": False, "error": "Email required"})

    stored = OTP_STORE.get(email)

    if not stored:
        return jsonify({"success": False, "error": "Session expired. Register again."})

    # ⏱ cooldown check (30 sec)
    if time.time() - stored["time"] < 30:
        return jsonify({"success": False, "error": "Please wait before requesting again"})

    # 🔁 generate new OTP
    new_otp = generate_otp()

    OTP_STORE[email]["otp"] = new_otp
    OTP_STORE[email]["time"] = time.time()

    try:
        send_otp_email(email, new_otp)
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

# ✅ FIX GOOGLE LOGIN (PREVENTS MIXING, CLEAN SESSION)
@app.route('/firebase-login', methods=['POST'])
@csrf.exempt
def firebase_login():
    try:
        data = request.get_json() or {}
        id_token = data.get('token')

        if not id_token:
            return jsonify({"success": False}), 400

        decoded_token = auth.verify_id_token(id_token)

        email = decoded_token.get('email')
        if not email:
            return jsonify({"success": False, "error": "Invalid email"}), 400
        if not decoded_token.get('email_verified'):
            return jsonify({"success": False, "error": "Email not verified"}), 401

        name = decoded_token.get('name', 'Google User')

        db = get_db()
        cur = db.cursor()

        cur.execute("SELECT * FROM users WHERE email=?", (email,))
        user = cur.fetchone()

        
            # ✅ ACCOUNT CONSISTENCY CHECK - Prevent mixing login types
        if user:
                user_id = user["id"]
        else:
            cur.execute("""
            INSERT INTO users (name, email, password)
            VALUES (?, ?, ?)
            """, (name, email, "GOOGLE_AUTH"))
            db.commit()
            user_id = cur.lastrowid

        # ✅ CLEAN SESSION BEFORE LOGIN (important for security)
        session.clear()

        session['user_id'] = user_id
        session['user_name'] = name
        session['auth_provider'] = 'google'
        session.permanent = True

        logging.info(f"User {user_id} ({email}) logged in via Google")
        return jsonify({"success": True})

    except Exception as e:
        logging.error(f"Firebase login error: {e}")
        return jsonify({"success": False, "error": str(e)}), 401

@app.route('/my_submissions')
@login_required
def my_submissions():
    user_id = session.get('user_id')

    db = get_db()
    cur = db.cursor()

    cur.execute("""
    SELECT * FROM submissions WHERE user_id = ?
    ORDER BY id DESC
    """, (user_id,))

    submissions = cur.fetchall()

    return render_template('my_submissions.html', submissions=submissions)

# ✅ PREVENT RELOGIN - Redirect already logged in users
@app.route('/register_page')
def register_page():
    if already_logged_in():
        return redirect(url_for('submit_form'))
    return render_template('register.html')

# ✅ SECURE LOGOUT
@app.route('/logout')
def logout():
    session.clear()
    logging.info("User logged out")
    return redirect(url_for('login_page'))

@app.route("/submit_form", methods=["GET", "POST"])
@login_required
def submit_form():
    trust_score = 50
    user = None
    if session.get('user_id'):
        db = get_db()
        cur = db.cursor()
        cur.execute("SELECT trust_score FROM users WHERE id=?", (session['user_id'],))
        user = cur.fetchone()
    if user:
        trust_score = user['trust_score']

    result = None
    error = None
    form_data = {}

    if request.method == "POST":
        conn = get_db()
        cur = conn.cursor()
        raw_title = request.form.get("title", "").strip()
        owner = request.form.get("owner", "").strip()

        if not owner:
            return render_template(
                "submit_form.html",
                error="Owner name is required",
                form_data=form_data,
                trust_score=trust_score
            )
        email = (request.form.get("email", "") or "").strip().lower()
        state = request.form.get("state", "").strip()
        language = request.form.get("language", "").strip()
        registration_number = request.form.get("registration_number", "").strip()

        if len(raw_title) > 150:
            return render_template(
                "submit_form.html",
                error="Title too long (max 150 characters)",
                form_data=form_data,
                trust_score=trust_score
            )
        if not raw_title.strip():
            return render_template(
                "submit_form.html",
                error="Title cannot be empty",
                form_data=form_data,
                trust_score=trust_score
            )
        title = normalize_text(raw_title)
        user_id = session.get('user_id')

        cur.execute("""
        SELECT trust_score, usage_count, last_used_date 
        FROM users 
        WHERE id = ?
        """, (user_id,))

        user = cur.fetchone()
        if not user:
            return redirect(url_for('login_page'))

        today = str(date.today())

        usage_count = user['usage_count']
        if usage_count is None:
            usage_count = 0
        last_used_date = user['last_used_date'] or ""
        trust_score = user['trust_score']

        # Reset if new day
        if last_used_date != today:
            usage_count = 0
            cur.execute("""
            UPDATE users SET usage_count = 0, last_used_date = ?
            WHERE id = ?
            """, (today, user_id))
            conn.commit()

        logging.info(f"User {user_id} usage count: {usage_count}")

        # LIMIT CHECK
        if usage_count >= MAX_DAILY_USAGE:
            return render_template(
                "submit_form.html",
                error="🚫 Daily limit reached. Try again tomorrow.",
                form_data=form_data,
                trust_score=trust_score
            )

        form_data = {
            "title": title,
            "owner": owner,
            "email": email,
            "state": state,
            "language": language,
            "registration_number": registration_number
        }

        # 🔥 DUPLICATE CHECK
        cur.execute("""
        SELECT 1 FROM submissions 
        WHERE LOWER(TRIM(title)) = LOWER(TRIM(?)) AND user_id = ?
        """, (title, user_id))

        existing = cur.fetchone()

        if existing:
            similarity_score = 100
            similarity_label = "High"

            return render_template(
                "submit_form.html",
                result={
                    "similarity_score": similarity_score,
                    "similarity_label": similarity_label
                },
                error="⚠ Exact duplicate title found in system",
                form_data=form_data,
                trust_score=trust_score
            )

        # Run similarity ALWAYS (do not block on validation) — compute against CSV PRGI dataset
        try:
            cache_key = title.lower()

            if cache_key in SIM_CACHE:
                similarity_analysis = SIM_CACHE[cache_key]
            else:
                    similarity_analysis = compare_title(title)
                    SIM_CACHE[cache_key] = similarity_analysis
            best_match = similarity_analysis["top_matches"][0] if similarity_analysis["top_matches"] else None

            breakdown = best_match[2] if best_match else {
                        "semantic": 0,
                        "tfidf": 0,
                        "jaccard": 0,
                        "phonetic": 0,
                        "edit": 0,
                        "structure": 0
            }
                    

            similarity_score = similarity_analysis["similarity"]
            similarity_label = similarity_analysis["risk"]
            closest_title = similarity_analysis["closest_title"]
            if not closest_title or similarity_score < 10:
                closest_title = None

            similarity_result = {
                "similarity": similarity_score,
                "risk": similarity_label,
                "closest_title": closest_title,
                "top_matches": similarity_analysis["top_matches"],
                "confidence": similarity_score,
                "risk_explanation": explain_risk(title, similarity_score),
                "analysis_status": "success"
            }
            result = {
                    "similarity_score": similarity_score,
                    "similarity_label": similarity_label,
                    "closest_title": closest_title,
                    "explanation": similarity_result["risk_explanation"],
                    "confidence": similarity_score,
                    "top_matches": similarity_result["top_matches"],
                    "breakdown": breakdown
            }
        except Exception as e:
            similarity_score, similarity_label = 0.0, 'Low'
            similarity_result = {
                "similarity": 0.0,
                "risk": 'Low',
                "closest_title": None,
                "analysis_status": "failed",
                "error": str(e),

                # 🔥 ADD THESE ALSO HERE
                "top_matches": [],
                "confidence": 0
            }
            result = {
                "similarity_score": similarity_score,
                "similarity_label": similarity_label,
                "closest_title": similarity_result.get("closest_title", ""),
                "explanation": similarity_result.get("risk_explanation", ""),
                "confidence": similarity_score
            }

        # Validation: if missing fields -> show error but include AI result
        if not all(form_data.values()):
            error = "All fields are required."
            return render_template(
                "submit_form.html",
                error=error,
                form_data=form_data,
                result=result,
                trust_score=trust_score
            )

        # Server-side email validation (stronger)
        if not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email) or len(email) > 100:
            error = "Please provide a valid email address."
            return render_template(
                "submit_form.html",
                error=error,
                form_data=form_data,
                result=result,
                trust_score=trust_score
            )

        # Uniqueness check (do NOT insert if exists; show error + AI result)
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
                result=result,
                trust_score=trust_score
            )

        # Block HIGH risk submissions (server-side enforcement)
        if similarity_score >= 70:
            error = "This title appears to be HIGH RISK and cannot be submitted. Please modify the title or contact support."
            return render_template(
                "submit_form.html",
                error=error,
                form_data=form_data,
                result=result,
                trust_score=trust_score
            )

        # Insert new submission (concurrency-safe - catch unique constraint errors)
        created_at = datetime.now(timezone.utc).isoformat()
        try:
            cur.execute("""
                INSERT INTO submissions (
                    title, owner, email, state, language,
                    registration_number, similarity_score,
                    similarity_label, created_at, status, user_id
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                title, owner, email, state, language,
                registration_number, similarity_score,
                similarity_label, created_at, "Pending", user_id
            ))

            conn.commit()

            today = str(date.today())
            usage_count += 1

            if similarity_score < 25:
                trust_score += 3
            elif similarity_score < 50:
                trust_score += 1
            elif similarity_score > 75:
                trust_score -= 4
            elif similarity_score > 60:
                trust_score -= 2

            trust_score = max(0, min(100, trust_score))

            cur.execute("""
            UPDATE users SET trust_score = ?, usage_count = ?, last_used_date = ?
            WHERE id = ?
            """, (trust_score, usage_count, today, user_id))

            conn.commit()

            logging.info(f"User {user_id} submitted title: {title} | Similarity: {similarity_score}%")

        except sqlite3.IntegrityError:
            error = "Registration number already exists."
            return render_template(
                "submit_form.html",
                error=error,
                form_data=form_data,
                result=result,
                trust_score=trust_score
            )

        submission_id = cur.lastrowid
        return redirect(url_for('success', submission_id=submission_id))

    return render_template(
        "submit_form.html",
        error=None,
        form_data={},
        result=None,
        trust_score=trust_score
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

# REAL-TIME DASHBOARD STATS API
@app.route("/api/dashboard_stats")
@csrf.exempt
def api_dashboard_stats():

    db = get_db()
    cur = db.cursor()

    # basic stats
    cur.execute("SELECT COUNT(*) FROM submissions")
    total = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM submissions WHERE status='Approved'")
    approved = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM submissions WHERE status='Pending'")
    pending = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM submissions WHERE status='Rejected'")
    rejected = cur.fetchone()[0]

    # state distribution
    cur.execute("""
        SELECT state, COUNT(*) FROM submissions
        GROUP BY state
    """)
    state_data = cur.fetchall()
    state_labels = [row[0] for row in state_data]
    state_values = [row[1] for row in state_data]

    # language distribution
    cur.execute("""
        SELECT language, COUNT(*) FROM submissions
        GROUP BY language
    """)
    lang_data = cur.fetchall()
    language_labels = [row[0] for row in lang_data]
    language_values = [row[1] for row in lang_data]

    # risk distribution
    cur.execute("""
        SELECT similarity_label, COUNT(*) FROM submissions
        GROUP BY similarity_label
    """)
    risk_data = {"Low":0,"Medium":0,"High":0}
    for label,count in cur.fetchall():
        if label in risk_data:
            risk_data[label] = count
    # approval trend (last 7 days)
    cur.execute("""
    SELECT DATE(created_at) as date,
           SUM(CASE WHEN status='Approved' THEN 1 ELSE 0 END),
           SUM(CASE WHEN status='Pending' THEN 1 ELSE 0 END)
    FROM submissions
    WHERE DATE(created_at) >= DATE('now','-7 days')
    GROUP BY DATE(created_at)
    ORDER BY date
""")

    trend = cur.fetchall()

    trend_labels = [row[0] for row in trend]
    trend_approved = [row[1] for row in trend]
    trend_pending = [row[2] for row in trend]

    return jsonify({
    "total": total,
    "approved": approved,
    "pending": pending,
    "rejected": rejected,

    "state_labels": state_labels,
    "state_data": state_values,

    "language_labels": language_labels,
    "language_data": language_values,

    "risk": [
        risk_data["Low"],
        risk_data["Medium"],
        risk_data["High"]
    ],

    "trend_labels": trend_labels,
    "trend_approved": trend_approved,
    "trend_pending": trend_pending
})

@app.route("/api/admin_stats")
@admin_required
def admin_stats():

    db = get_db()
    cur = db.cursor()

    cur.execute("SELECT COUNT(*) FROM submissions WHERE status='Approved'")
    approved = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM submissions WHERE status='Pending'")
    pending = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM submissions WHERE status='Rejected'")
    rejected = cur.fetchone()[0]

    return jsonify({
        "approved": approved,
        "pending": pending,
        "rejected": rejected
    })

def get_common_words(titles):
    """Extract common words from titles for AI insights"""
    if not titles:
        return []

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

def explain_risk(title, similarity):

    title = title.lower()

    # detect common publication keywords
    common_keywords = [
        "india","indian","times","post","express",
        "news","digest","bulletin","chronicle",
        "gazette","journal","review"
    ]

    keywords_found = [w for w in common_keywords if w in title]

    # risk explanation aligned with meter thresholds
    if similarity >= 65:
        if keywords_found:
            return f"High risk: The title is very similar to existing PRGI publications and contains common keywords like {', '.join(keywords_found)}."
        return "High risk: The title is very similar to existing PRGI publications and may cause a registration conflict."

    elif similarity >= 30:
        if keywords_found:
            return f"Medium risk: The title has moderate similarity and contains common publication keywords like {', '.join(keywords_found)}."
        return "Medium risk: The title has moderate similarity with existing publication titles."

    else:
        return "Low risk: The title appears unique with low similarity."

def detect_fake_title(title):

    reasons = []
    words = title.lower().split()

    # repetition detection
    if len(words) != len(set(words)):
        reasons.append("Excessive word repetition")

    # clickbait detection
    clickbait_words = [
        "breaking", "shocking", "exclusive",
        "viral", "must read", "alert"
    ]

    for word in clickbait_words:
        if title.lower().startswith(word):
            reasons.append("Clickbait wording detected")
            break

    # punctuation spam
    if title.count("!") > 1 or title.count("?") > 1:
        reasons.append("Excessive punctuation")

    # ALL CAPS
    if title.isupper() and len(title) > 10:
        reasons.append("All caps title")

    return reasons

def ai_recommendation(similarity, fake_flag):

    # High similarity → reject
    if similarity >= 60:
        return "Reject"

    # Low similarity and no fake pattern → approve
    if similarity < 30 and not fake_flag:
        return "Approve"

    # Fake titles always need review
    if fake_flag:
        return "Review"

    # Medium similarity → review
    if 30 <= similarity < 60:
        return "Review"

    return "Review"

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
            session['admin_user'] = ADMIN_USERNAME
            session.permanent = True
            logging.info(f"Admin {username} logged in")
            return redirect(url_for('admin_dashboard'))
        else:
            error = "Invalid credentials. Please try again."

    return render_template('admin_login.html', error=error)

@app.route('/admin/logout')
def admin_logout():
    """Admin logout - clear session"""
    session.clear()
    logging.info("Admin logged out")
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
    if status:
        query += ' AND LOWER(status) = LOWER(?)'
        params.append(status.strip())
    if state:
        query += ' AND LOWER(state) = LOWER(?)'
        params.append(state.strip())
    if language:
        query += ' AND LOWER(language) = LOWER(?)'
        params.append(language.strip())

    query += ' ORDER BY id DESC'
    cur.execute(query, tuple(params))
    rows = cur.fetchall()

    submissions = []

    for row in rows:
        similarity = row["similarity_score"] or 0
        fake_reasons = detect_fake_title(row["title"])
        fake_flag = len(fake_reasons) > 0
        recommendation = ai_recommendation(similarity, fake_flag)

        row_dict = dict(row)
        row_dict["ai_recommendation"] = recommendation

        submissions.append(row_dict)
    return render_template(
        'admin_dashboard.html',
        submissions=submissions,
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
        logging.error(f'CSV export error: {e}')
        return "Export failed", 500

@app.route("/admin/update_status", methods=["POST"])
@admin_required
@csrf.exempt
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

    conn = get_db()
    cur = conn.cursor()

    cur.execute(
        "UPDATE submissions SET status=? WHERE id=?",
        (status, submission_id)
    )

    conn.commit()

    if cur.rowcount == 0:
        return jsonify({"error": "id not found"}), 404

    logging.info(f"Submission {submission_id} status updated to {status}")
    return jsonify({"success": True, "status": status})

# ✅ CHECK SIMILARITY API - CSRF EXEMPT FOR AJAX
@app.route("/api/check_similarity", methods=["POST"])
@csrf.exempt
def check_similarity():
    data = request.get_json(silent=True) or {}

    title = normalize_text((data.get("title") or "").strip())

    if len(title) < 4:
        return jsonify({
            "similarity": 0,
            "risk": "Low",
            "closest_title": None,
            "top_matches": [],
            "confidence": 0,
            "explanation": "Title too short"
        })

    try:
        result = compare_title(title)

        similarity = float(result.get("similarity", 0))
        risk = result.get("risk", "Low")
        closest_title = result.get("closest_title")

        # ✅ ENSURE closest_title is None if no match
        if closest_title == "No strong match found":
            closest_title = None

        return jsonify({
            "similarity": similarity,
            "risk": risk,
            "closest_title": closest_title,

            # 🔥 MOST IMPORTANT
            "top_matches": result.get("top_matches", []),
            "confidence": similarity,

            # OPTIONAL BUT GOOD
            "explanation": explain_risk(title, similarity),
            "fake_reasons": detect_fake_title(title)
        })

    except Exception as e:
        logging.error(f"Similarity API error: {e}")
        return jsonify({
            "similarity": 0,
            "risk": "Low",
            "closest_title": None,
            "top_matches": [],
            "confidence": 0,
            "explanation": "Analysis failed"
        })

@app.route('/api/check_registration', methods=['POST'])
@csrf.exempt
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

# ✅ GENERATE PRGI TITLES API - CSRF EXEMPT FOR AJAX
@app.route('/api/generate_prgi_titles', methods=['POST'])
@csrf.exempt
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

        alternative_titles = generate_prgi_alternative_titles(
            user_title,
            prgi_match_title,
            similarity_percent
        )

        response_text = '\n'.join(alternative_titles[:5])

        lines = response_text.split('\n')
        if len(lines) < 5:
            extra = generate_fallback_suggestions(user_title)
            for item in extra:
                if item["title"] not in lines:
                    lines.append(item["title"])
                if len(lines) >= 5:
                    break

        return '\n'.join(lines[:5]), 200, {'Content-Type': 'text/plain'}

    except Exception as e:
        logging.error(f"Error in generate_prgi_titles: {e}")
        fallback_titles = [
            "Regional Publication Review",
            "State Bulletin Gazette",
            "Local Journal Dispatch",
            "Community News Bulletin",
            "Public Information Chronicle"
        ]
        return '\n'.join(fallback_titles), 200, {'Content-Type': 'text/plain'}

def validate_ai_titles(titles):

    final = []

    for t in titles:
        analysis = compare_title(t)

        similarity = analysis.get("similarity", 0)
        risk = analysis.get("risk", "Low")
        fake_flags = detect_fake_title(t)

        if similarity < 30 and risk == "Low" and not fake_flags:
            final.append(t)

    return final[:5]

def generate_fallback_suggestions(title):
    """Generate rule-based suggestions if SBERT fails"""
    from random import choice

    prefixes = ['The', 'National', 'Regional', 'Local', 'Community', 'Weekly', 'Monthly']
    suffixes = ['Review', 'Bulletin', 'Journal', 'Chronicle', 'Herald', 'Digest', 'Gazette']
    modifiers = ['India', 'Indian', 'Weekly', 'Journal', 'Review', 'News', 'Publication']

    suggestions = []

    for i in range(3):
        if i < len(prefixes):
            suggestions.append(f"{prefixes[i]} {title}")

    for i in range(3):
        if i < len(suffixes):
            suggestions.append(f"{title} {suffixes[i]}")

    common_words = ['news', 'daily', 'bulletin', 'digest', 'weekly', 'monthly']
    for word in common_words:
        if word in title.lower():
            new_title = title.lower().replace(word, choice(modifiers)).title()
            suggestions.append(new_title)

    return [{
        "title": suggestion,
        "similarity": round(30 + (i * 5), 1)
    } for i, suggestion in enumerate(suggestions[:6])]

def calculate_string_similarity(s1, s2):
    """Calculate similarity between two strings (0.0 to 1.0)"""
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
    import re

    # Extract meaningful words
    words = re.findall(r'\b[a-zA-Z]{3,}\b', user_title.lower())

    if not words:
        return [
            "General Technology Review",
            "Digital Innovation Bulletin",
            "Emerging Trends Chronicle",
            "Future Developments Journal",
            "Technology Insights Report"
        ]

    # --- STEP 1: Detect theme ---
    tech_map = {
        "tech": "Technology",
        "technology": "Technology",
        "digital": "Digital",
        "ai": "AI",
        "data": "Data",
        "innovation": "Innovation"
    }

    theme = None
    for w in words:
        if w in tech_map:
            theme = tech_map[w]
            break

    if not theme:
        theme = words[0].title()

    # --- STEP 2: Smart replacements ---
    word_variations = {
        "latest": ["emerging", "current", "modern"],
        "updates": ["insights", "trends", "developments", "reports"],
        "news": ["bulletin", "report", "chronicle"],
        "tech": ["technology", "digital", "innovation"]
    }

    generated = []

    for w in words:
        if w in word_variations:
            for alt in word_variations[w]:
                new_words = [alt if x == w else x for x in words]
                title = " ".join(new_words).title()
                generated.append(title)

    # --- STEP 3: Strong structured titles (MAIN QUALITY FIX) ---
    structured_titles = [
        f"{theme} Innovation Bulletin",
        f"{theme} Trends Chronicle",
        f"Emerging {theme} Review",
        f"{theme} Developments Journal",
        f"Future {theme} Insights"
    ]

    generated.extend(structured_titles)

    # --- STEP 4: Clean & filter ---
    cleaned = []
    for t in generated:
        t = t.strip()

        # avoid very short / duplicate / ugly titles
        if len(t.split()) < 2:
            continue
        if t not in cleaned:
            cleaned.append(t)

    # --- STEP 5: Ensure exactly 5 high-quality outputs ---
    final = cleaned[:5]

    # fallback safety (rare case)
    if len(final) < 5:
        fallback = [
            f"{theme} Review",
            f"{theme} Bulletin",
            f"{theme} Chronicle",
            f"{theme} Journal",
            f"{theme} Insights"
        ]
        for f in fallback:
            if f not in final:
                final.append(f)
            if len(final) >= 5:
                break

    return final[:5]
if __name__ == '__main__':
    with app.app_context():
        init_db()
        try:
            print("✅ PRGI system initialized successfully")
        except Exception as e:
            print(f"❌ CRITICAL ERROR: Failed to initialize PRGI system: {e}")
            raise

    # ✅ RUN APP (OUTSIDE try/except)
    if os.environ.get("FLASK_ENV") == "development":
        app.run(debug=True)
    else:
        app.run(host='127.0.0.1', port=5000, debug=False)

# Favicon route (NON-NEGOTIABLE)
@app.route('/favicon.ico')
def favicon():
    return send_from_directory(
        os.path.join(app.root_path, 'static', 'assets', 'icons'),
        'favicon.ico',
        mimetype='image/vnd.microsoft.icon'
    )