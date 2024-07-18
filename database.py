# database.py
import sqlite3

def get_db_connection():
    conn = sqlite3.connect('sunshine-movies.db')
    conn.row_factory = sqlite3.Row  # This allows us to access columns by name
    return conn

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            user_id INTEGER PRIMARY KEY,
            username TEXT UNIQUE,
            password TEXT,
            avatar TEXT
        )
    ''')

    user_id = 0
    cursor.execute('INSERT or IGNORE INTO users(user_id, username, password, avatar) VALUES (?, ?, ?, ?)', (user_id, 'admin', 'admin', 'https://api.dicebear.com/9.x/adventurer/svg?seed=1'))

    conn.commit()
    
    conn.close()



def view_users():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users")
    rows = cursor.fetchall()
    conn.close()
    for row in rows:
        print(dict(row))

# Run this function to initialize the database
if __name__ == "__main__":
     view_users()