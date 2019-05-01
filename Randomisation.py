import pandas as pd
import numpy as np
import pickle
import random
import copy
from RegressionRF import RegressionModel
import openpyxl

def generateVOCScreenings(vocDf2013,vocDf2015, sliceDf, matchedMovies):
    screeningList = list()
    prevStartIndex = 0
    startIndex = 0
    vocDf = vocDf2013
    for index in range(0, len(matchedMovies)):
        
        if startIndex == 371: #the 2015 df starts at this index
            vocDf = vocDf2015
        
        startIndex = sliceDf.loc[index]['start']
        endIndex = sliceDf.loc[index]['end']
        screening = vocDf.loc[startIndex:endIndex,:]
        screeningList.append(screening)
        
        prevStartIndex = startIndex
        
    return screeningList

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
def emptyScreenings(screenings, randomisedScreenings, matchedMovies):
    screeningList = list()
    randomScreeningList = list()
    movieList = list()
    for screeningIndex in range(0, len(screenings)):
        if not(np.isnan(screenings[screeningIndex].values).all()):
            screeningList.append(screenings[screeningIndex])
            randomScreeningList.append(randomisedScreenings[screeningIndex])
            movieList.append(matchedMovies[screeningIndex])
    return screeningList,randomScreeningList,movieList

#the randomisedScreenings have NaN instances within the 
def replaceNaNInRandomisedScreenings(randomisedScreenings,vocList):
    for screeningIndex in range(0, len(randomisedScreenings)):
        if np.isnan(randomisedScreenings[screeningIndex].values).any():
            #the screening has some NaN instances then replace with random values
            for vocIndex in range(0, len(randomisedScreenings[screeningIndex].values)):
                voc = randomisedScreenings[screeningIndex].values[vocIndex]
                if np.isnan(voc[0]):
                    randomIndex = random.randint(0,len(vocList+1)) #generate a random number
                    while np.isnan(vocList[randomIndex]):
                        #continue generating random numbers if NaN was returned 
                        randomIndex = random.randint(0,len(vocList+1))

                    randomisedScreenings[screeningIndex].values[vocIndex] = vocList[randomIndex]

    return randomisedScreenings

def main():
    
    #read in the various csvs
    #2013 Dataset
    vocPath = 'Numerical Data/2013VOCData.csv'
    voc2013DfAll = pd.read_csv(vocPath, header = 0, nrows = 74208, low_memory=False)
    movieScreeningsPath = 'Numerical Data/screening_times.csv'
    movingScreeningsDf = pd.read_csv(movieScreeningsPath, usecols = ['scheduled','movie','filled %'])
    movieRuntimesPath = 'Numerical Data/movie_runtimes.csv'
    movieRuntimeDf = pd.read_csv(movieRuntimesPath, usecols = ['movie', 'runtime (mins)', 'effective runtime'])
    #2015 Dataset
    starWarsPath = 'Numerical Data/Star Wars-The Force Awakens.csv'
    starWarsScreeningDf = pd.read_csv(starWarsPath)
    imOffThenPath = 'Numerical Data/I\'m Off Then.csv'
    imOffThenScreeningDf = pd.read_csv(imOffThenPath)
    helpIShrunkTheTeacherPath = 'Numerical Data/Help, I Shrunk My Teacher.csv'
    helpIShrunkTheTeacherScreeningDf = pd.read_csv(helpIShrunkTheTeacherPath)
    vocPath = 'Numerical Data/2015VOCData.csv'
    voc2015DfAll = pd.read_csv(vocPath)
    #remove first column of 2015 voc df as its not used
    voc2015DfAll.drop("Unnamed: 0", axis=1, inplace=True) 

    #import co2Slice pickle objects
    slicePath = 'Pickle Objects/CO2SliceDict.p'
    sliceDict = pickle.load(open(slicePath, "rb" )) #contains df of co2 slice indices and matched movie list
    
    #user macros
    voc = 'CO2'
    vocSave = False
    modelSave = False

    #results df
    resultsHeader = ['RandomState','VOC','RMSE', 'MAE', 'R2']
    resultsList = list()
    
    for voc in voc2015DfAll.columns:
        if voc == 'Time':
            continue
        elif voc == 'm18.0338':
            break
        else:
            try:
                indexMask = list(voc2013DfAll.columns).index(voc)
            except ValueError: #the voc isnt within the 2013 VOC dataset
                continue 
            print(voc)
            #create normal voc screening list
            vocDf2013 = voc2013DfAll.loc[:,[voc]]
            vocDf2015 = voc2015DfAll.loc[:,[voc]]
            screeningList = generateVOCScreenings(vocDf2013,vocDf2015, sliceDict['sliceDf'], sliceDict['matchedMovies'])
            #create randomised voc list
            voc2013RandomisedList = copy.deepcopy(list(vocDf2013[voc]))
            voc2015RandomisedList = copy.deepcopy(list(vocDf2015[voc]))
            random.shuffle(voc2013RandomisedList)
            random.shuffle(voc2015RandomisedList)
            vocDf2013Randomised = pd.DataFrame.from_dict({voc:voc2013RandomisedList})
            vocDf2015Randomised = pd.DataFrame.from_dict({voc:voc2015RandomisedList})
            randomisedScreeningList = generateVOCScreenings(vocDf2013Randomised, vocDf2015Randomised, sliceDict['sliceDf'], sliceDict['matchedMovies'])
            #remove empty screenings from the list
            matchedMovies = copy.deepcopy(sliceDict['matchedMovies'])
            screeningList, randomisedScreeningList, matchedMovies = emptyScreenings(screeningList, randomisedScreeningList, matchedMovies)
            #randomised screenings will have some NaN instances within then so replace those NaN instances with randomly selected VOC
            entireVocList = np.append(vocDf2013.values, vocDf2015.values)
            randomisedScreeningList = replaceNaNInRandomisedScreenings(randomisedScreeningList,entireVocList)
            #perform normalisation
            screeningList = normalisation(screeningList, voc)
            randomisedScreeningList = normalisation(randomisedScreeningList, voc)

            vocScreeningDict = {'screenings':screeningList, 'matchedMovies':matchedMovies}
            vocRandomisedScreeningDict = {'screenings':randomisedScreeningList, 'matchedMovies':matchedMovies}

            RMSE,MAE,R2 = RegressionModel(vocScreeningDict, modelSave,False,False, voc)
            resultsList.append([False, voc, RMSE,MAE,R2])
            RMSE,MAE,R2 = RegressionModel(vocRandomisedScreeningDict, modelSave,False,False, voc)
            resultsList.append([True, voc, RMSE,MAE,R2])
            print()

    #create results Df
    resultsDf = pd.DataFrame(resultsList,columns=resultsHeader)
    #write df to output file
    resultsDf.to_excel("results.xlsx") 
    resultsDf.to_csv('results.csv', sep=',', encoding='utf-8')

main()