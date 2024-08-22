from typing import List
from fastapi import FastAPI, HTTPException, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from trivial_recommender import Trivial
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


@app.get("/models")
async def models(model: str, ratings: str = Query(...), users: str = Query(...), movies: str = Query(...)):

    ratings = json.loads(ratings)
    users = json.loads(users)
    movies = json.loads(movies)
    model = json.loads(model)

    df_ratings = []

    # create a dataframe from the ratings list
    for i in range(len(ratings)):
        df_ratings.append([users[i], movies[i], ratings[i]])
    
    df = pd.DataFrame(df_ratings, columns=["userId", "movieId", "rating"])


    # Start computations for the selected model
    recommendation_movieId = []
    if model == "Trivial":
        trivial = Trivial(df)
        recommendation_movieId = trivial.topMovieTrivial
    print(recommendation_movieId)
    return recommendation_movieId
    return {"model": model}