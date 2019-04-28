import pandas as pd
import numpy as np
import pickle
import random
import copy
import openpyxl
import RegressionRF

def generateVOCScreenings(vocDf2013,vocDf2015, sliceDf, matchedMovies):
    screeningList = list()
    prevStartIndex = 0
    startIndex = 0
    vocDf = vocDf2013
    for index in range(0, len(matchedMovies)):
        
        if prevStartIndex > startIndex: #switch to 2015 dataframe
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

def main():

    #user macros    
    modelSave = False

    #results df
    resultsHeader = ['RandomState','VOC','RMSE', 'MAE', 'R2']
    resultsList = list()

    #read in the various csvs
    #2013 Dataset
    vocPath = 'Numerical Data/2013VOCData.csv'
    voc2013DfAll = pd.read_csv(vocPath, header = 0, nrows = 74208)
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

    for voc in voc2013DfAll.columns:
        if voc == 'Time':
            continue
        else:
            try:
                indexMask = list(voc2015DfAll.columns).index(voc)
            except ValueError: #the voc isnt within the 2015 df
                continue 
            
            #if the voc is within the 2015VocDf create the entire vocDf
            vocDf2013 = voc2013DfAll.loc[:,[voc]]
            vocDf2015 = voc2015DfAll.loc[:,[voc]]
            #create randomised vocDf
            voc2013RandomisedList = copy.deepcopy(list(vocDf2013['CO2']))
            voc2015RandomisedList = copy.deepcopy(list(vocDf2015['CO2']))
            random.shuffle(voc2013RandomisedList)
            random.shuffle(voc2015RandomisedList)
            vocDf2013Randomised = pd.DataFrame.from_dict({voc:voc2013RandomisedList})
            vocDf2015Randomised = pd.DataFrame.from_dict({voc:voc2015RandomisedList})
            screeningList = generateVOCScreenings(vocDf2013,vocDf2015, sliceDict['sliceDf'], sliceDict['matchedMovies'])
            randomisedScreeningList = generateVOCScreenings(vocDf2013Randomised, vocDf2015Randomised, sliceDict['sliceDf'], sliceDict['matchedMovies'])
            #create normalised screening and normalised random screening list
            screeningList = normalisation(screeningList, voc)
            randomisedScreeningList = normalisation(randomisedScreeningList, voc)
            
            vocScreeningDict = {'screenings':screeningList, 'matchedMovies':sliceDict['matchedMovies']}
            vocRandomisedScreeningDict = {'screenings':randomisedScreeningList, 'matchedMovies':sliceDict['matchedMovies']}
            
            RMSE,MAE,R2 = RegressionModel(vocScreeningDict, modelSave)
            resultsList.append([False, voc, RMSE,MAE,R2])
            RMSE,MAE,R2 = RegressionModel(vocRandomisedScreeningDict, modelSave)
            resultsList.append([True, voc, RMSE,MAE,R2])
            
    #create results Df
    resultsDf = pd.DataFrame(resultsList, columns=resultsHeader)
    resultsDf.to_excel("results.xlsx") 
    resultsDf.to_csv('results.csv, sep=',', encoding='utf-8')   

main()