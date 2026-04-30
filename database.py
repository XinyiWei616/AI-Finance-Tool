import sqlite3

DB_NAME = "database.db"

def create_table():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()

    c.execute('''
    CREATE TABLE IF NOT EXISTS transactions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        amount REAL,
        category TEXT,
        date TEXT,
        description TEXT
    )
    ''')

    conn.commit()
    conn.close()

