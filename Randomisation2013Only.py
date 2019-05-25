import pandas as pd
import numpy as np
import pickle
import random
import copy
from RegressionRF import RegressionModel
import openpyxl
from math import trunc

def generateVOCScreenings(vocDf2013, sliceDf, matchedMovies):
    screeningList = list()
    prevStartIndex = 0
    startIndex = 0
    vocDf = vocDf2013
    for index in range(0, len(matchedMovies)):
        
        startIndex = sliceDf.loc[index]['start']
        endIndex = sliceDf.loc[index]['end']
        
        if startIndex == 371: #the 2015 df starts at this index
            return screeningList #return as there is no 2015 data being considered
        
        screening = pd.DataFrame(vocDf.iloc[startIndex:endIndex+1])
        screeningList.append(screening)
        prevStartIndex = startIndex
        
#normalise the screenings by the max value
def normalisation(vocScreenings, voc):
    normalisedVOCList = list()
    for screening in vocScreenings:
        normalisedVOCFrame = copy.deepcopy(screening)
        normalisedVOCFrame = normalisedVOCFrame.values/max(screening.values)
        normalisedVOCFrame = normalisedVOCFrame.flatten()
        normalisedScreening= pd.DataFrame.from_dict({voc:normalisedVOCFrame})
        normalisedVOCList.append(normalisedScreening)
    return normalisedVOCList

#some vocs dont have the recorded screenings so remove then
#then remove those same screenings from the randomisedScreeningList
def removeNaNScreenings(screenings, randomisedScreenings, matchedMovies):
    screeningList = list()
    randomScreeningList = list()
    movieList = list()
    for screeningIndex in range(0, len(screenings)):
        if not(np.isnan(screenings[screeningIndex].values).any()):
            screeningList.append(screenings[screeningIndex])
            randomScreeningList.append(randomisedScreenings[screeningIndex])
            movieList.append(matchedMovies[screeningIndex])
    return screeningList,randomScreeningList,movieList

#the randomisedScreenings have NaN instances within the 
def replaceNaNInRandomisedScreenings(randomisedScreeningList,entireVocList):
    for screeningIndex in range(0, len(randomisedScreeningList)):
        if (np.isnan(randomisedScreeningList[screeningIndex].values)).any():
            for vocIndex in range(0, len(randomisedScreeningList[screeningIndex].values)):
                voc = randomisedScreeningList[screeningIndex].values[vocIndex]
                if np.isnan(voc[0]):
                    randomIndex = random.randint(0,len(entireVocList)-1)
                    while np.isnan(entireVocList[randomIndex]):
                            #continue generating random numbers if NaN was returned 
                            randomIndex = random.randint(0,len(entireVocList)-1)
                    randomisedScreeningList[screeningIndex].values[vocIndex] = entireVocList[randomIndex]
    return randomisedScreeningList


def main():
    
    #read in the various csvs
    #2013 Dataset
    vocPath = 'Numerical Data/2013VOCData.csv'
    voc2013Df = pd.read_csv(vocPath, header = 0, nrows = 74208, low_memory=False)
    #note that the 2013 data has 430 measured vocs
    movieScreeningsPath = 'Numerical Data/screening_times.csv'
    movingScreeningsDf = pd.read_csv(movieScreeningsPath, usecols = ['scheduled','movie','filled %'])
    movieRuntimesPath = 'Numerical Data/movie_runtimes.csv'
    movieRuntimeDf = pd.read_csv(movieRuntimesPath, usecols = ['movie', 'runtime (mins)', 'effective runtime'])
    #import co2Slice pickle objects
    slicePath = 'Pickle Objects/CO2SliceDict.p'
    sliceDict = pickle.load(open(slicePath, "rb" )) #contains df of co2 slice indices and matched movie list
    
    #user macros
    vocSave = False
    modelSave = False
    randomisationIterations = 100

    #results df
    resultsHeader = ['RandomState','VOC','RMSE', 'MAE', 'R2']
    resultsList = list()
    
    for vocIndex in range(378,len(voc2013Df.columns)): #allows for running specific vocs through the randomisation process
        voc = voc2013Df.columns[vocIndex]
        if voc == 'Time':
            continue
        else:
            print(voc)
            resultsList = list()
            
            for i in range(0,randomisationIterations):
                #create normal voc screening list
                vocDf = voc2013Df.loc[:,voc]
                #generate voc screenings
                screeningList = generateVOCScreenings(vocDf, sliceDict['sliceDf'], sliceDict['matchedMovies'])
                matchedMovies = copy.deepcopy(sliceDict['matchedMovies'])
                #normalise the screenings
                screeningList = normalisation(screeningList, voc)
                #create randomised voc list
                voc2013RandomisedList = copy.deepcopy(list(voc2013Df[voc]))
                random.shuffle(voc2013RandomisedList)
                vocDf2013Randomised = pd.DataFrame.from_dict({voc:voc2013RandomisedList})
                randomisedScreeningList = generateVOCScreenings(vocDf2013Randomised, sliceDict['sliceDf'], sliceDict['matchedMovies'])
                #remove screenings from the normal voc screening list with any NaNs within them
                screeningList, randomisedScreeningList, matchedMovies = removeNaNScreenings(screeningList, randomisedScreeningList, matchedMovies)
                #replace any of the remaining NaN's within the randomised screening list (sampling without replacement)
                randomisedScreeningList = replaceNaNInRandomisedScreenings(randomisedScreeningList,vocDf.values)
                #normalise the randomised screening list
                randomisedScreeningList = normalisation(randomisedScreeningList, voc)     
                
                #create randomised and unrandomised voc dictionary 
                vocScreeningDict = {'screenings':screeningList, 'matchedMovies':matchedMovies}
                vocRandomisedScreeningDict = {'screenings':randomisedScreeningList, 'matchedMovies':matchedMovies}

                #run RF regression on randomised and unrandomised vocs 
                RMSE,MAE,R2 = RegressionModel(vocScreeningDict, modelSave,False,False, voc)
                resultsList.append([False, voc, RMSE,MAE,R2])
                RMSE,MAE,R2 = RegressionModel(vocRandomisedScreeningDict, modelSave,False,False, voc)
                resultsList.append([True, voc, RMSE,MAE,R2])

        #create results Df
        resultsDf = pd.DataFrame(resultsList,columns=resultsHeader)
        #write df to output file
        resultsPath = str(voc) + '.csv'
        resultsDf.to_csv(resultsPath, sep=',', encoding='utf-8')

main()