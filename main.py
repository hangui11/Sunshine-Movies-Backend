from fastapi import FastAPI, HTTPException, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sqlite3
from database import get_db_connection, get_max_user_id
import numpy as np

import pandas as pd
import json
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"message": "Hello World"}


class User(BaseModel):
    user_id: int
    username: str
    password: str
    avatar: str

    class Config:
        orm_mode = True

@app.get("/users/", response_model=User)
async def get_user(username: str = Query(..., alias="username")):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
    row = cursor.fetchone()
    conn.close()
    if row is None:
        raise HTTPException(status_code=404, detail="User not found")
    return {"user_id": row["user_id"], "username": row["username"], "password": row["password"] ,"avatar": row["avatar"]}

@app.post("/user/")
async def post_user(username: str = Query(..., alias='username'),
                    password: str = Query(..., alias='password')):
    
    # Generar avatar aleatorio
    avatar = 'https://api.dicebear.com/9.x/adventurer/svg?seed=' + str(np.random.randint(100))
    
    # Establecer conexión a la base de datos
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # Verificar si el username ya existe en la base de datos
        cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
        existing_user = cursor.fetchone()
        
        if existing_user:
            # Si el usuario ya existe, manejar el error o actualizar el usuario según sea necesario
            conn.close()
            raise HTTPException(status_code=400, detail="Username already exists")
        
        user_id = get_max_user_id(cursor)+1
        # Insertar nuevo usuario en la base de datos
        cursor.execute('INSERT INTO users(user_id, username, password, avatar) VALUES (?, ?, ?, ?)',
                       (user_id, username, password, avatar))
        
        # Commit de la transacción
        conn.commit()
        
        # Cerrar conexión a la base de datos
        conn.close()
        
        return {'message': 'User created successfully'}
    
    except sqlite3.Error as e:
        # En caso de error de SQLite, realizar rollback y manejar la excepción
        conn.rollback()
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    
    except Exception as e:
        # Manejar otras excepciones inesperadas
        conn.rollback()
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")
