import numpy as np
import pandas as pd
from math import trunc
import pickle
import copy
import random
import xgboost as xgb
from sklearn import metrics
from sklearn import model_selection
import os
from sklearn.preprocessing import MinMaxScaler

#load and save the feature/label dataframes and save into a dictionary
def loadFeaturesLabels(vocs_index,features_index):
    files = os.listdir("data//mounted//Windowed Features & Label Csvs//")

    features = list()
    vocs = list()
    matchedMovies = list()

    for file in files:
        movieName = file.split("-")[:-3][0]
        df = pd.read_csv("data//mounted//Windowed Features & Label Csvs//" + file)
        df.drop("Unnamed: 0", axis=1, inplace=True)

        vocs.append(df.iloc[:,vocs_index])
        features.append(df.iloc[:,features_index])
        matchedMovies.append(movieName)
        
    return vocs,features,matchedMovies

def train_test_split(labels,features,matchedMovies,trainingMovies,testMovie):
    #create training and test df
    testDf = pd.DataFrame([])
    trainDf = pd.DataFrame([])
    for voc, movie_features, movie in zip(labels,features,matchedMovies):
        #attached features and labels
        infoDf = movie_features.join(voc)
        #drop any NaN's 
        infoDf.dropna(inplace=True)
        #scale the labels between 0 and 1
        sc = MinMaxScaler()
        try:
            infoDf.iloc[:,-1] = sc.fit_transform(infoDf.iloc[:,-1].values.reshape(-1,1))
        except ValueError:
            #screening is empty (all NaN)
            continue

        if movie in testMovie:
            testDf = pd.concat([testDf,infoDf], ignore_index=True)
        else:
            trainDf = pd.concat([trainDf,infoDf], ignore_index=True)
            
    return trainDf,testDf


def main():
    
    random.seed(106)
    
    saveURL = 'data//mounted//Results//'
    
    movieRuntimesPath = 'data/mounted/Numerical Data/movie_runtimes.csv'
    movieRuntimeDf = pd.read_csv(movieRuntimesPath, usecols = ['movie', 'runtime (mins)', 'effective runtime'])
    movieList = list(movieRuntimeDf['movie'])

    vocs_index = range(3180, 3683)
    features_index=range(0,3180)
    number_of_vocs = 503
    
    vocs,features,matchedMovies = loadFeaturesLabels(vocs_index,features_index)
    
    for voc_index in range(200, 300):
    
        R2_score = list()
        RMSE_score = list()
        testingMovies = list()
        Random_R2_score = list()
        Random_RMSE_score = list()

        #extract voc index
        labels = [screening.iloc[:,voc_index] for screening in vocs]
        voc = labels[0].name
        
        print(voc)
        
        for iter_no in range(0, 5): 
            
            print(iter_no)

            #iterate through all movies using movies as test  movie
            for movie in movieList:

                #train test split
                trainingMovies = list(movieRuntimeDf['movie'])
                testMovie = [movie]
                trainingMovies.pop(trainingMovies.index(testMovie[0]))
                
                
                trainingFeatures,testingFeatures = train_test_split(labels,features,matchedMovies,trainingMovies,testMovie)
             
                if (trainingFeatures.shape[0] != 0 and testingFeatures.shape[0] != 0):
                    trainingLabels = trainingFeatures.iloc[:,-1]
                    trainingFeatures.drop(trainingFeatures.columns[-1], axis=1, inplace=True)
                    testingLabels = testingFeatures.iloc[:,-1]
                    testingFeatures.drop(testingFeatures.columns[-1], axis=1, inplace=True)

                    #run experiment
                    #normal 
                    print('Train normal model')
                    regressor = xgb.XGBRegressor(n_estimators=100, n_jobs=-1)
                    regressor.fit(trainingFeatures, trainingLabels.ravel())
                    #predict
                    predictions = regressor.predict(testingFeatures)
                    r2_score = metrics.r2_score(testingLabels, predictions)
                    rmse = np.sqrt(metrics.mean_squared_error(testingLabels,predictions))
                    #print and save
                    print("R2 score:", r2_score)
                    print("RMSE: ", rmse)
                    R2_score.append(r2_score)
                    RMSE_score.append(rmse)
                    testingMovies.append(movie)

                    #shuffle the labels in the training set and use the same test set
                    np.random.shuffle(trainingLabels)
                    trainingFeatures.drop(trainingFeatures.columns[-1], axis=1, inplace=True)

                    #random
                    print('Train randomised model')
                    regressor = xgb.XGBRegressor(n_estimators=100, n_jobs=-1)
                    regressor.fit(trainingFeatures, trainingLabels.ravel())
                    #predict
                    predictions = regressor.predict(testingFeatures)
                    r2_score = metrics.r2_score(testingLabels, predictions)
                    rmse = np.sqrt(metrics.mean_squared_error(testingLabels,predictions))
                    #print and save
                    print("Random R2 score:", r2_score)
                    print("Random RMSE: ", rmse)
                    Random_R2_score.append(r2_score)
                    Random_RMSE_score.append(rmse)


        #create and output a dataframe 
        pd.DataFrame({'RMSE':RMSE_score, 
                      'R2 Score':R2_score, 
                      'Random RMSE': Random_RMSE_score,
                      'Random R2 Score':Random_R2_score,
                      'Test Movie': testingMovies}).to_csv(saveURL + voc + ".csv")


main()


