#!/usr/bin/env python3
"""Verify database seeding."""

import sqlite3
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, 'database', 'database.db')

conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

# Get stats
cur.execute("SELECT COUNT(*) FROM submissions")
total = cur.fetchone()[0]

cur.execute("SELECT COUNT(*) FROM submissions WHERE status='Approved'")
approved = cur.fetchone()[0]

# Show approved titles
cur.execute("SELECT title, status FROM submissions WHERE status='Approved' ORDER BY created_at DESC LIMIT 15")
rows = cur.fetchall()

print(f"📊 Database Statistics:")
print(f"   Total submissions: {total}")
print(f"   Approved: {approved}")
print(f"\n✅ Approved Titles (for learning engine):")
for i, (title, status) in enumerate(rows, 1):
    print(f"   {i:2d}. {title}")

conn.close()
