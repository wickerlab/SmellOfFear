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

def processAudio(runtime, audioFeatures):
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

def processSubtitles(subs):
    header = ['sentiment', 'sentiment value']
    subSentimentDf = pd.DataFrame(columns=header)
    for sentimentIndex in range(0, len(subs)):
        sentiment = subs[sentimentIndex]
        if len(sentiment) != 0:
            subSentimentDf.loc[sentimentIndex] = [sentiment['sentiment'], sentiment['sentimentValue']]
        else:
            subSentimentDf.loc[sentimentIndex] = [np.NaN, np.NaN]
    
    return subSentimentDf


def main():

    #user macros
    windowedVOCs = False
    lengthOfWindow = 10

    #import vocs
    vocDict = pickle.load(open("Pickle Objects/normalisedVOC.p", "rb" )) #dictionary object that contains the vocs

    #import movie runtimes
    movieRuntimesPath = 'Numerical Data/movie_runtimes.csv'
    movieRuntimeDf = pd.read_csv(movieRuntimesPath, usecols = ['movie', 'runtime (mins)', 'effective runtime'])
    movieList = list(movieRuntimeDf['movie'])

    #overall feature and labels df
    featureDf = pd.DataFrame([]) #film feature dataframe
    labelDf = pd.DataFrame([]) #voc dataframe
  
    #import pickle objects for movies and then assemble the dataframes  
    for movie in movieList:

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
        sentimentDf = processSubtitles(sentiment)

        inputDf = pd.concat([colourDf,shadeDf,audioDf,sentimentDf], axis = 1)

        #output df
        screenings = vocDict[movie]
        #create overall input and output df
        for i in range(0, len(screenings)):
            featureDf = pd.concat([featureDf,inputDf])
            if not(windowedVOCs):
                screening = screenings[i]['CO2']
                labelDf = pd.concat([labelDf, screening['CO2']])
            else:
                #using windowed VOCs
                header = ['VOC' + str(x) for x in range(1,lengthOfWindow+1)]
                vocWindowDf = pd.DataFrame(columns = header)
                for index in range(0, len(screening)):
                    vocWindow = screening[index]['CO2'].values
                    vocWindowDf.loc[index] = vocWindow 
                labelDf = pd.concat([labelDf, vocWindowDf])

    #change header label on output header
    labelDf.columns = ['CO2']

    #train test split
    #create training and test datasets
    featuresTrain, featuresTest, labelsTrain, labelsTest = train_test_split(featureDf, labelDf, test_size= 0.20) #80 20 train test split
    #second train test split is to randomly remove screenings to test them seperately

    #regression model
    regressor = RandomForestRegressor(n_estimators=10000, random_state=0)
    regressor.fit(featuresTrain, labelsTrain)
    labelsPred = regressor.predict(featuresTest)

    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(labelsTest, labelsPred))) 
    











main()
