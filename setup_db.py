"""
Run this ONCE before starting the app to create the student database.
    python setup_db.py
"""

import sqlite3

def create_database():
    conn = sqlite3.connect("student.db")
    cur = conn.cursor()

    # Create table
    cur.execute("DROP TABLE IF EXISTS STUDENT")
    cur.execute("""
        CREATE TABLE STUDENT (
            NAME    TEXT,
            CLASS   TEXT,
            SECTION TEXT
        )
    """)

    # Sample data
    students = [
        ("Aarav Shah",       "Data Science",  "A"),
        ("Priya Patel",      "Data Science",  "B"),
        ("Rohan Mehta",      "Math",          "A"),
        ("Sneha Desai",      "Math",          "B"),
        ("Amit Kumar",       "Physics",       "A"),
        ("Anjali Verma",     "Physics",       "C"),
        ("Vikram Singh",     "Data Science",  "A"),
        ("Pooja Sharma",     "Chemistry",     "B"),
        ("Kiran Joshi",      "Math",          "C"),
        ("Neha Gupta",       "Data Science",  "C"),
        ("Arjun Nair",       "Physics",       "B"),
        ("Divya Rao",        "Chemistry",     "A"),
        ("Suresh Iyer",      "Math",          "A"),
        ("Meena Pillai",     "Data Science",  "B"),
        ("Rahul Tiwari",     "Chemistry",     "C"),
    ]

    cur.executemany("INSERT INTO STUDENT VALUES (?, ?, ?)", students)
    conn.commit()
    conn.close()
    print(f"✅ student.db created with {len(students)} students.")

if __name__ == "__main__":
    create_database()