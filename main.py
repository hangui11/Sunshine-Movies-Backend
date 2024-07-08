from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

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


@app.get("/users")
async def root():
    df = pd.read_csv('../users.csv')
    data = []
    for i, row in df.iterrows():
        userId = row['userId']
        userName = row['userName']
        password = row['password']
        role = row['role']
        data.append({'userId': userId, 'username': userName, 'password': password, 'role': role})
    return data