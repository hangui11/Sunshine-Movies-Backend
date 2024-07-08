import sqlite3


conn = sqlite3.connect('sunshine-movies.db')

cursor = conn.cursor()

cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
               user_id INTEGER PRIMARY KEY)

'''
)

user_id = 3
cursor.execute('INSERT INTO users(user_id) VALUES (?)', (user_id,))

conn.commit()
cursor.execute("SELECT * FROM users")
result = cursor.fetchall()

print(result)

conn.close()
