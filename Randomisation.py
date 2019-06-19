import pandas as pd
import numpy as np
import pickle
import random
import copy
from RegressionRF import RegressionModel
import openpyxl
from math import trunc

def generateVOCScreenings(vocDf2013,vocDf2015, sliceDf, matchedMovies):
    screeningList = list()
    prevStartIndex = 0
    startIndex = 0
    vocDf = vocDf2013
    for index in range(0, len(matchedMovies)):
        
        startIndex = sliceDf.loc[index]['start']
        endIndex = sliceDf.loc[index]['end']
        if startIndex == 371: #the 2015 df starts at this index
            vocDf = vocDf2015
        screening = pd.DataFrame(vocDf.iloc[startIndex:endIndex+1,0])
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

def normalisationStandardScaler(vocScreenings, voc):
    normalisedVOCList = list()
    for screening in vocScreenings:
        scaler = StandardScaler()
        scaler.transform(screening)
        normalisedVOCFrame = normalisedVOCFrame.values/max(screening.values)
        normalisedVOCFrame = normalisedVOCFrame.flatten()
        normalisedScreening= pd.DataFrame.from_dict({voc:normalisedVOCFrame})
        normalisedVOCList.append(normalisedScreening)
    return normalisedVOCList    

#some vocs have NaN measurements during the decided screening times. Ignore these screenings
#also remove empty screenings
def removeNaNScreenings(screenings, randomisedScreenings, matchedMovies):
    screeningList = list()
    randomScreeningList = list()
    movieList = list()
    for screeningIndex in range(0, len(screenings)):
        if not(np.isnan(screenings[screeningIndex].values).any()) and len(screenings[screeningIndex].values) != 0:
            screeningList.append(screenings[screeningIndex])
            randomScreeningList.append(randomisedScreenings[screeningIndex])
            movieList.append(matchedMovies[screeningIndex])
    return screeningList,randomScreeningList,movieList



#column header matching issue between 2013 and 2015 
def vocRounding(vocDf):
    vocList = list()
    for index in range(0, len(vocDf.columns)):
        if vocDf.columns[index] == 'Time' or vocDf.columns[index] == 'ocs' or vocDf.columns[index] == 'co' or vocDf.columns[index] == 'CO2':
            vocList.append(vocDf.columns[index])    
        else:
            #string slice to get the molar mass
            voc = vocDf.columns[index]
            mass = (trunc(float(voc[1:])*1000))/1000 #TRUNCATE TO 3DP
            vocList.append(mass)
    return vocList

#generate randomised screenings
def createRandomisedVOCScreenings(vocRandomised, runtimeList, movieList ,matchedMovies):
    startIndex = 0
    screeningList = list()
    for movie in matchedMovies:
        try:
            runtime = runtimeList[movieList.index(movie)]
        except ValueError:
            continue
        endIndex = startIndex + runtime
        screening = vocRandomised[startIndex:endIndex]
        screeningList.append(screening)
        startIndex = endIndex
    return screeningList

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
    vocSave = False
    modelSave = False
    randomisationIterations = 100

    #results df
    resultsHeader = ['RandomState','VOC','RMSE', 'MAE', 'R2']
    
    voc2015Col = vocRounding(voc2015DfAll)
    voc2013Col = vocRounding(voc2013DfAll)
    voc2013Df = copy.deepcopy(voc2013DfAll)
    voc2015Df = copy.deepcopy(voc2015DfAll)
    voc2013Df.columns = voc2013Col
    voc2015Df.columns = voc2015Col

    for vocIndex in range(75,100): #allows to run seperate vocs on seperate cpu cores
        voc = voc2015Df.columns[vocIndex]
        if voc == 'Time':
            continue
        else:
            try:
                indexMask = list(voc2013Df.columns).index(voc)
            except ValueError: #the voc isnt within the 2013 VOC dataset
                continue 

            print(voc)
            resultsList = list()
            
            for i in range(0,randomisationIterations):
                #create normal voc screening list
                vocDf2013 = voc2013Df.iloc[:,[indexMask]]
                vocDf2015 = voc2015Df.iloc[:,[vocIndex]]   

                #generate screenings
                screeningList = generateVOCScreenings(vocDf2013,vocDf2015, sliceDict['sliceDf'], sliceDict['matchedMovies'])
                matchedMovies = copy.deepcopy(sliceDict['matchedMovies'])
                #use logical vectors to remove all NaNs and create randomised voc lists
                voc2013RandomisedList= vocDf2013.values[np.logical_not(np.isnan(vocDf2013.values))]
                voc2015RandomisedList = vocDf2015.values[np.logical_not(np.isnan(vocDf2015.values))]
                vocRandomised = np.append(voc2013RandomisedList,voc2015RandomisedList, axis=0)
                random.shuffle(vocRandomised)
                #generate randomised screenings
                randomisedScreenings = createRandomisedVOCScreenings(vocRandomised, list(movieRuntimeDf['effective runtime']), list(movieRuntimeDf['movie']) ,matchedMovies)
                randomisedScreeningList = list()
                list(map(lambda screening : randomisedScreeningList.append(pd.DataFrame.from_dict({voc:screening})), randomisedScreenings))
                #remove normal screenings with NaN values in the screenings
                screeningList, randomisedScreeningList, matchedMovies = removeNaNScreenings(screeningList, randomisedScreeningList, matchedMovies)
                #normalise both screenings 
                screeningList = normalisation(screeningList, voc)
                randomisedScreeningList = normalisation(randomisedScreeningList, voc)
                #remove any leftover NaN screenings
                screeningList, randomisedScreeningList, matchedMovies = removeNaNScreenings(screeningList, randomisedScreeningList, matchedMovies)
                #create randomised and unrandomised list
                vocScreeningDict = {'screenings':screeningList, 'matchedMovies':matchedMovies}
                vocRandomisedScreeningDict = {'screenings':randomisedScreeningList, 'matchedMovies':matchedMovies}

                RMSE,MAE,R2 = RegressionModel(vocScreeningDict, voc)
                resultsList.append([False, voc, RMSE,MAE,R2])
                RMSE,MAE,R2 = RegressionModel(vocRandomisedScreeningDict, voc)
                resultsList.append([True, voc, RMSE,MAE,R2])

        #create results Df
        resultsDf = pd.DataFrame(resultsList,columns=resultsHeader)
        #write df to output file
        resultsPath = str(voc) + '.csv'
        resultsDf.to_csv(resultsPath, sep=',', encoding='utf-8')

main()