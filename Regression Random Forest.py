import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import copy

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

def calculateDeltaFilmFeatures(movieFeatureDf, featureHeader):
    deltaFeatureDf = pd.DataFrame(columns=featureHeader)
    for rowIndex in range(0,movieFeatureDf.shape[0]-1):
        tempDf = pd.concat([movieFeatureDf.loc[rowIndex],movieFeatureDf.loc[rowIndex+1]]).values
        deltaFeatureDf.loc[rowIndex]= tempDf
    return deltaFeatureDf


def main():

    #overall feature and labels df
    featureDf = pd.DataFrame([]) #film feature dataframe
    labelDf = pd.DataFrame([]) #voc dataframe

    #user macros
    deltaVOCs = False
    windowedVOCs = True
    lengthOfWindow = 10
    
    #import vocs
    if not(deltaVOCs) and not(windowedVOCs):
        vocDict = pickle.load(open("Pickle Objects/normalisedScreeningsDict.p", "rb" ))
    elif not(deltaVOCs) and windowedVOCs:
        vocDict = pickle.load(open("Pickle Objects/normalisedWindowedScreeningsDict.p", "rb" ))
    elif deltaVOCs and not(windowedVOCs):
        vocDict = pickle.load(open("Pickle Objects/deltaScreeningsDict.p", "rb" ))
    else:
        print('WRONG COMBINATION OF MACROS')
        return

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

            runtime = movieRuntimeDf.loc[movieList.index(movie)]['effective runtime']
            colourDf = processVisuals(colour, runtime, True)
            shadeDf = processVisuals(shade, runtime, False)
            audioDf = processAudio(runtime, audio)
            sentimentDf = processSubtitles(sentiment,runtime)

            inputDf = pd.concat([colourDf,shadeDf,audioDf,sentimentDf], axis = 1)
            movieFeatureDict[movie] = inputDf
        except FileNotFoundError:
            print(movie)
    
    #remove all screenings of im off then and help i shrunk the teacher as at the current time do not have the movies
    vocDict = removeMovies(vocDict)
    
    #create label and feature df

    for i in range(0, len(vocDict['screenings'])): 

        matchedMovie = vocDict['matchedMovies'][i]

        if not(deltaVOCs):
            featureDf = pd.concat([featureDf, movieFeatureDict[matchedMovie]])
        else:
            featureHeader = list(movieFeatureDict[matchedMovie].columns) + list(movieFeatureDict[matchedMovie].columns)
            deltaDf = calculateDeltaFilmFeatures(movieFeatureDict[matchedMovie],featureHeader)
            featureDf = pd.concat([featureDf, deltaDf])

        if not(windowedVOCs):
            screening = vocDict['screenings'][i]
            labelDf = pd.concat([labelDf, screening['CO2']])
        else:
            screening = vocDict['screenings'][i]
            #using windowedVOCsed VOCs
            header = ['VOC' + str(x) for x in range(1,lengthOfWindow+1)]
            vocWindowDf = pd.DataFrame(columns = header)
            for index in range(0, len(screening)):
                vocWindow = screening[index]['CO2'].values
                vocWindowDf.loc[index] = vocWindow 
            labelDf = pd.concat([labelDf, vocWindowDf])

    #relabel column title 
    if not(windowedVOCs):
        labelDf.columns = ['VOC']


     
    #train test split
    #create training and test datasets
    print('Train Test Split')
    featuresTrain, featuresTest, labelsTrain, labelsTest = train_test_split(featureDf, labelDf, test_size= 0.20) #80 20 train test split
    #second train test split is to randomly remove screenings to test them seperately

    #regression model
    print('Using Window ' + str(windowedVOCs))
    print('Train Model')
    regressor = RandomForestRegressor(n_estimators=10000, random_state=0)
    if not(windowedVOCs):
        regressor.fit(featuresTrain, labelsTrain.values.ravel())
    else:
        regressor.fit(featuresTrain, labelsTrain)
        
    print('Test Model')
    labelsPred = regressor.predict(featuresTest)

    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(labelsTest, labelsPred))) 
    

main()
