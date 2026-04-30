import sqlite3
from database import DB_NAME

def get_connection():
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    return conn 

def insert_transaction(amount, category, date, description):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute('''
    INSERT INTO transactions (amount, category, date, description)
    VALUES (?, ?, ?, ?)
    ''', (amount, category, date, description))

    conn.commit()
    conn.close()

def get_transactions():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute('SELECT * FROM transactions')
    rows = cursor.fetchall()

    conn.close()

    return [dict(row) for row in rows] 
