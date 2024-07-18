from fastapi import FastAPI, HTTPException, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sqlite3
from database import get_db_connection, init_db

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
def get_user(user_id: int = Query(..., alias="user_id")):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE user_id = ?", (user_id,))
    row = cursor.fetchone()
    conn.close()
    if row is None:
        raise HTTPException(status_code=404, detail="User not found")
    return {"user_id": row["user_id"], "username": row["username"], "password": row["password"] ,"avatar": row["avatar"]}
