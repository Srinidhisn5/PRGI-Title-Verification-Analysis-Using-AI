#!/usr/bin/env python3
"""
Seed the database with APPROVED titles to train the learning engine.
This gives the engine good examples of what "safe" PRGI titles look like.
"""

import os
import sqlite3
from datetime import datetime, timezone

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, 'database', 'database.db')

# Good PRGI-style approved titles (technology/news themed)
APPROVED_TITLES = [
    "Hindustan Times",
    "India Today",
    "The Hindu",
    "Indian Express",
    "Desh Ka News",
    "Rashtriya Bulletin",
    "Jan Varta",
    "Lok Patrika",
    "Nav Bharat Herald",
    "Technology Chronicle",
    "Digital Innovation Times",
    "Emerging Trends Digest",
    "Tech News Bulletin",
    "Innovation Report",
    "Rashtriya Samachar Patrika",
]

def seed_database():
    """Insert approved titles into the database."""
    try:
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        
        created_at = datetime.now(timezone.utc).isoformat()
        
        inserted = 0
        skipped = 0
        
        for title in APPROVED_TITLES:
            # Check if already exists
            cur.execute(
                "SELECT 1 FROM submissions WHERE LOWER(title) = LOWER(?)",
                (title,)
            )
            if cur.fetchone():
                print(f"⏭️  Skipping (exists): {title}")
                skipped += 1
                continue
            
            # Insert as approved
            try:
                cur.execute("""
                    INSERT INTO submissions (
                        title, owner, email, state, language,
                        registration_number, similarity_score,
                        similarity_label, created_at, status, user_id
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    title,
                    "System",
                    "system@prgi.gov.in",
                    "National",
                    "English",
                    f"SEED-{inserted}",
                    0.0,
                    "Low",
                    created_at,
                    "Approved",  # ✅ KEY: Mark as Approved for training
                    None
                ))
                inserted += 1
                print(f"✅ Inserted: {title}")
            except sqlite3.IntegrityError as e:
                print(f"❌ Error on {title}: {e}")
                skipped += 1
        
        conn.commit()
        conn.close()
        
        print(f"\n{'='*60}")
        print(f"✅ Seed complete: {inserted} titles inserted, {skipped} skipped")
        print(f"{'='*60}")
        print("\n📚 Learning engine will rebuild on next submission.")
        print("   This approved data trains it to recognize better titles.")
        
    except Exception as e:
        print(f"❌ Database error: {e}")
        return False
    
    return True

if __name__ == "__main__":
    seed_database()
