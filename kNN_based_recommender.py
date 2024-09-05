import numpy as np

from item_based_recommender import item_based_recommender as item
from user_based_recommender import user_based_recommender as user

class KnnHybrid:
    '''
    This method initializes the KnnHybrid class with the necessary data and parameters,
    preparing it for the KNN hybrid algorithm.
     - self.topK: Stores the number of top recommendations to be generated
     - self.movies: Stores a dataframe with the movies information
    '''
    def __init__(self, movies, k=5) -> None:
        self.movies = movies
        self.topK = k

    '''
    This function merges the top recommendations from the user-based and item-based recommendation algorithms,
    using a combination metric that combines the product of the two similarities and the Euclidean distance between them.
    '''
    def mergeUsersAndItemsRecommendations(self, usersRecommendations, itemsRecommendations):
        ## method 1: using pandas
        # recommenderUsers = pd.DataFrame(usersRecommendations, columns=['movieId', 'similarityUser'])
        # recommenderItems = pd.DataFrame(itemsRecommendations, columns=['movieId', 'similarityItem'])

        # recommenderHybrid = recommenderUsers.merge(recommenderItems, how='inner', on='movieId')
        # recommenderHybrid['productSimilarities'] = recommenderHybrid['similarityUser'] * recommenderHybrid['similarityItem']
        # recommenderHybrid['maxSimilarity'] = recommenderHybrid[['similarityUser', 'similarityItem']].max(axis=1)
        # recommenderHybrid['euclidianDistance'] = ((recommenderHybrid['similarityUser'] - recommenderHybrid['similarityItem']) **2) **0.5

        # recommendations1 = [tuple(x) for x in recommenderHybrid[['movieId', 'productSimilarities']].to_records(index=False)]
        # recommendations2 = [tuple(x) for x in recommenderHybrid[['movieId', 'maxSimilarity']].to_records(index=False)]
        # recommendations3 = [tuple(x) for x in recommenderHybrid[['movieId', 'euclidianDistance']].to_records(index=False)]
        
        ## method 2: using dict
        recommenderUsers = {movieId: similarity for movieId, similarity in usersRecommendations}
        recommenderItems = {movieId: similarity for movieId, similarity in itemsRecommendations}

        common_movies = set(recommenderUsers.keys()) & set(recommenderItems.keys())

        # productSimilarities = []
        # euclidianDistance = []
        
        # The combination capture both difference and the relationship between the two similarities
        # Since the combination can favor the complexity of this methods because it capture a wider 
        # range of the relations between elements.
        combinationProductAndEuclidian = []
        for i in common_movies:
            userRecommend = recommenderUsers[i]
            itemRecommend = recommenderItems[i]
            # productSimilarities.append((i, userRecommend * itemRecommend))
            # euclidianDistance.append((i, np.sqrt((userRecommend - itemRecommend)**2) ))
            combinationProductAndEuclidian.append((i, (userRecommend * itemRecommend) * (np.sqrt((userRecommend - itemRecommend)**2))))

        self.recommendations = sorted(combinationProductAndEuclidian, key=lambda x:x[1], reverse=True)
        return self.recommendations
    
   