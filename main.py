from fastapi import FastAPI, HTTPException, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from trivial_recommender import Trivial
from user_based_recommender import UserToUser
from item_based_recommender import ItemToItem
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
async def models(model: str, user_id: str, user_size: str, ratings: str = Query(...), users: str = Query(...), movies: str = Query(...)):

    ratings = json.loads(ratings)
    users = json.loads(users)
    movies = json.loads(movies)
    model = json.loads(model)
    user_id = json.loads(user_id)
    user_size = json.loads(user_size)

    user_ids = [i+1 for i in range(user_size-1)]
    print(user_ids)
    

    if user_id == 0: return {"message": "This is a admin user"}

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
    if model == "User-to-User":
        u2u = UserToUser(df, user_ids)
        recommendation_movieId = u2u.user_based_recommender(user_id)
    if model == 'Item-to-Item':
        i2i = ItemToItem(df, movies, user_ids)
        recommendation_movieId = i2i.item_based_recommender(user_id)
    if model == 'K-Nearest-Neighbor':
        pass
    print(recommendation_movieId)
    return {"result": recommendation_movieId}