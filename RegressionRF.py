import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import copy
from sklearn.externals import joblib
import random

import DataPipeline

#frames were collected at 1/3fps so for a 30 second period there are 10 frames. This function just groups the 
#dominant frame colour or shade components to within their respective intervals
def grouping(visualList):
    movieVisuals = list()
    for index in range(0, int(len(visualList)/10)):
        segment = visualList[index*10:index*10+10]
        movieVisuals.append(segment)
    return movieVisuals

def processVisuals(movieVisualData, runtime, isColour):
    visualDataIntervals = grouping(movieVisualData)
    #the visual data also has the credits accounted for so remove them
    visualDataIntervals = visualDataIntervals[:runtime]
    #create a dataframe 
    if isColour: 
        #create a dominant colour dataframe
        framesPerInterval = 10
        header = list();
        for i in range(1,framesPerInterval+1):
            header = header + ['R'+str(i), 'G' + str(i),  'B'+str(i)]
    else: #shade object to be parsed
        framesPerInterval = 10
        header = ['S' + str(x) for x in range(1,framesPerInterval+1)]
    
    visualDf = pd.DataFrame(columns=header)
    #assemble the dataframe
    for segment in visualDataIntervals:
        index = visualDataIntervals.index(segment)
        colourRow = list()
        for colour in segment:
            if isColour:
                colourRow = colourRow + [colour[0], colour[1], colour[2]]
            else:
                colourRow = colourRow + [colour[0]]
        #assign that colour row to the dataframe
        visualDf.loc[index] = colourRow
            
    return visualDf

def processAudio(runtime, audio):
    audioFeatures = list(audio.keys())

    audioDf = pd.DataFrame(columns=[])        
    for key in audioFeatures:
        audio[key] = audio[key][:runtime]

        #assemble df 
        #create header
        if key != 'tempo':
            header = [key + str(x) for x in range(1, len(audio[key][0])+1)]
        else:
            header = ['tempo']

        audioFeatureDf = pd.DataFrame(columns=header)
        for index in range(0, len(audio[key])):
            feature = audio[key][index]
            audioFeatureDf.loc[index] = feature

        #concatenate featureDf to audioDf
        audioDf = pd.concat([audioDf,audioFeatureDf], axis=1)
    
    return audioDf

def processSubtitles(subs, effectiveRuntime):
    
    header = ['sentiment value']
    subSentimentDf = pd.DataFrame(columns=header)
    for sentimentIndex in range(0, len(subs)):
        sentiment = subs[sentimentIndex]
        if len(sentiment) != 0:
            if sentiment['sentimentValue'] == np.NaN:
                print('YES')
            else:         
                subSentimentDf.loc[sentimentIndex] = [sentiment['sentimentValue']]
        else:
            subSentimentDf.loc[sentimentIndex] = [-1] #indicates no dialog occurred during the scene
        
        #enforce no dialog until the credit scene if there is in fact no dialog
        if len(subSentimentDf) != effectiveRuntime:
            #no dialog at the end thus need to fill the rest with -1
            for index in range(0, effectiveRuntime-len(subSentimentDf)+1):
                 subSentimentDf.loc[index] = [-1]
    
    return subSentimentDf

def processASL(asl, effectiveRuntime):
    
    header = ['average shot length']
    aslDf = pd.DataFrame(columns=header)
    for index in range(0, effectiveRuntime): 
        aslValue = asl[index]
        aslDf.loc[index] = aslValue
    
    return aslDf

def removeMovies(vocDict):

    #remove all screenings of im off then and help i shrunk the teacher as at the current time do not have the movies
    screenings = list()
    matchedMovies = list()
    for movieIndex in range(0, len(vocDict['matchedMovies'])):
        movie = vocDict['matchedMovies'][movieIndex]
        if movie != "Help, I Shrunk My Teacher" and movie != "I'm Off Then":
            #add good screenings to a modified screening list
            matchedMovies.append(movie)
            screenings.append(vocDict['screenings'][movieIndex])
    #replace
    vocDict = dict()
    vocDict['matchedMovies'] = matchedMovies
    vocDict['screenings'] = screenings
    
    return vocDict

def createTrainingAndTestSet(vocDict):

    #80:20 train:test, thus randomly allocate 80% of screenings to test and 20% to test
    numberOfScreenings = len(vocDict['screenings'])
    testScreeningList = list()
    testMovieList = list()
    
    #create test set
    for screeningNumber in range(0,round(0.2*numberOfScreenings)):
        randomIndex = random.randint(0, len(vocDict['screenings'])-1)
        screening = vocDict['screenings'].pop(randomIndex)
        testScreeningList.append(screening)
        matchedMovie = vocDict['matchedMovies'].pop(randomIndex)
        testMovieList.append(matchedMovie)
    
    #create training and test dict
    testingDict = {'screenings':testScreeningList,'matchedMovies':testMovieList}
    trainingDict = {'screenings':vocDict['screenings'],'matchedMovies':vocDict['matchedMovies']}
    
    return testingDict,trainingDict

def createInputOutputDf(vocDict, movieFeatureDict, voc):
    featureDf = pd.DataFrame([]) #film feature dataframe
    labelArray = np.array([])
    for i in range(0, len(vocDict['screenings'])): 
        matchedMovie = vocDict['matchedMovies'][i]
        featureDf = pd.concat([featureDf, movieFeatureDict[matchedMovie]])
        screening = vocDict['screenings'][i][voc]
        labelArray = np.append(labelArray, screening.values)
    labelDf = pd.DataFrame(labelArray) #voc dataframe
    labelDf.columns = ['VOC']
    return featureDf, labelDf

def RegressionModel(vocDict, voc):

    #overall feature and labels df
    featureDf = pd.DataFrame([]) #film feature dataframe
    labelDf = pd.DataFrame([]) #voc dataframe
    
    #import movie runtimes
    movieRuntimesPath = 'Numerical Data/movie_runtimes.csv'
    movieRuntimeDf = pd.read_csv(movieRuntimesPath, usecols = ['movie', 'runtime (mins)', 'effective runtime'])
    movieList = list(movieRuntimeDf['movie'])

    movieFeatureDict = dict() #dict contains the movie film features with the keys being the movies
    #import pickle objects for movies and then assemble the dataframes  
    for movie in movieList:
        try:
            #load pickle feauture objects
            featurePath = 'Pickle Objects/Audio Feature Pickle Objects/' + movie + '.p'
            audio = pickle.load(open(featurePath, "rb" )) 
            featurePath = 'Pickle Objects/Colour Pickle Objects/' + movie + '.p'
            colour = pickle.load(open(featurePath, "rb" )) 
            featurePath = 'Pickle Objects/Shade Pickle Objects/' + movie + '.p'
            shade = pickle.load(open(featurePath, "rb" )) 
            featurePath = 'Pickle Objects/Subtitle Sentiment Pickle Objects/' + movie + '.p'
            sentiment = pickle.load(open(featurePath, "rb" )) 
            featurePath = 'Pickle Objects/ASL Pickle Objects/' + movie + '.p'
            asl = pickle.load(open(featurePath, "rb" )) 
            
            runtime = int(movieRuntimeDf.loc[movieList.index(movie)]['effective runtime'])
            colourDf = processVisuals(colour, runtime, True)
            shadeDf = processVisuals(shade, runtime, False)
            audioDf = processAudio(runtime, audio)
            sentimentDf = processSubtitles(sentiment,runtime)
            aslDf = processASL(asl, runtime)
        
            inputDf = pd.concat([colourDf,shadeDf,audioDf,sentimentDf,aslDf], axis = 1)
            movieFeatureDict[movie] = inputDf
            
        except FileNotFoundError:
            pass
            
    print('Finished Loading Film Features')
    
    #remove all screenings of im off then and help i shrunk the teacher as at the current time do not have the movies
    vocDict = removeMovies(vocDict)
    print('Train Test Split')
    testingDict,trainingDict = createTrainingAndTestSet(vocDict)
    print('Creating Feature & Label Dataframes')
    featuresTrain, labelsTrain = createInputOutputDf(trainingDict,movieFeatureDict, voc)
    featuresTest, labelsTest = createInputOutputDf(testingDict,movieFeatureDict, voc)

    #regression model
    print('Train Model')
    regressor = RandomForestRegressor() #random forest will base parameters
    regressor.fit(featuresTrain, labelsTrain.values.ravel())

    print('Test Model')
    labelsPred = regressor.predict(featuresTest)
    
    RMSE = np.sqrt(metrics.mean_squared_error(labelsTest, labelsPred))
    MAE = metrics.mean_absolute_error(labelsTest, labelsPred)
    R2 = metrics.r2_score(labelsTest, labelsPred)
                   
    print('Root Mean Squared Error: ', RMSE) 
    print('Absolute Mean Error: ', MAE)
    print('R Squared: ', R2)
    
    return RMSE,MAE,R2
    
