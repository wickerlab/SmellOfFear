import numpy as np
import pandas as pd
from math import trunc
import pickle
import copy
import random
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,precision_score

#vocRounding is used to truncate the VOCs to 3dp allowing for flexible matching between 2013 and 2015 datasets
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

#generate normalised screenings
#remove invalid screenings (divide by NaN or divide by 0)
#scale vocs between 0 and 1
def generateNormalisedScreenings(sliceDict, vocData):
    screeningList = list()
    matchedMovies = list()
    for index in range(0,sliceDict['sliceDf'].shape[0]):
        start,end = sliceDict['sliceDf'].loc[index]
        screening = vocData[start:end+1]
        normalisedFrame = copy.deepcopy(screening)
        if max(normalisedFrame.values) != 0 and not(np.isnan(max(normalisedFrame.values))):
            normalisedFrame = normalisedFrame.values.reshape(-1,1)
            scaler = MinMaxScaler()
            normalisedFrame = scaler.fit_transform(normalisedFrame)
            normalisedFrame = np.transpose(normalisedFrame)
            screeningList.append(normalisedFrame)
            matchedMovies.append(sliceDict['matchedMovies'][index])
    return screeningList, matchedMovies

#train test split - one movie is left out for the test set 
def movieTrainTestSplit(movieList,matchedMovies,screeningList):
    
    testScreeningList = list()
    testMovieList = list()
    testMovie = movieList[random.randint(0, len(movieList)-1)] #pick random test movie 
    while True:
        try:
            matchedIndex = matchedMovies.index(testMovie)
            screening = screeningList.pop(matchedIndex)
            testScreeningList.append(screening)
            matchedMovie = matchedMovies.pop(matchedIndex)
            testMovieList.append(testMovie)
        except ValueError:
            break
    
    return testScreeningList,testMovieList,screeningList,matchedMovies


def inputOutputDf(screeningList):
    
    #input - 9 vocs
    #output - 1 voc 
    
    deltaDf = np.array([])
    inputDf = np.array([])
    info = np.array([])
    for screening in screeningList:
        
        #Calculate deltas
        deltas = np.array([screening[0,x+1] - screening[0,x] for x in range(8,screening.shape[1]-1)])
#         if deltaDf.size == 0:
#             deltaDf = deltas
#         else:
#             deltaDf = np.append(deltaDf,deltas)
        
        #AR VOC Input
        screening = screening[0] #remove the 1st dimension
        listOfInstances = [screening[x:x+9] for x in range(0,len(screening))]
        inputDf = np.array([])
        for instance in listOfInstances:
            if len(instance) == 9:
                if inputDf.size == 0:
                    inputDf = instance
                else:
                    inputDf = np.vstack((inputDf,instance))

        #add labels to the output
        #cut off last row (as cannot predict it)
        inputDf = inputDf[:-1]

        #get categorical values
        # 1 up
        # 0 no change
        # -1 down
        deltaLabels = categoricalValues(deltas)
        
        #connect the input to output
        if info.size == 0: 
            info = np.hstack((inputDf,np.expand_dims(deltaLabels, axis=1)))
        else:
            instance = np.hstack((inputDf,np.expand_dims(deltaLabels, axis=1)))
            info = np.vstack((info,instance))

    return info

def categoricalValues(deltas):
    #sort and need to abs to find the 10% value
    sortedDeltas = np.sort(np.abs(deltas))
    tenthPercentile = np.percentile(sortedDeltas, 10)

    # +- tenthPercentile is the no change category 
    # greater than tenthPercentile is the going up category 
    # less than tenthPercentile is the going down category
    #categories - 'no change', 'up', 'down'
    #up = 1
    #down = -1
    #no change = 0

    changeGroundTruth = np.zeros(deltas.shape[0])
    #up category
    upMask = np.greater_equal(deltas, tenthPercentile)
    changeGroundTruth[upMask] = 1
    #down category
    downMask = np.less_equal(deltas, tenthPercentile)
    changeGroundTruth[downMask] = -1
    
    return changeGroundTruth

def ClassificationModel(featuresTrain,labelsTrain, labelsTest,featuresTest):
    print('Train Model')
    clf = RandomForestClassifier()
    clf.fit(featuresTrain, labelsTrain)  
    
    print('Test Model')
    predictedLabels = clf.predict(featuresTest)
    
    #compute accuracy and precision
    precisionScore = precision_score(labelsTest, predictedLabels)
    accuracyScore = accuracy_score(labelsTest, predictedLabels)
    
    return precisionScore,accuracyScore

def main():

    #import various numeric csvs
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

    #full list of movies
    movieList = list(movieRuntimeDf['movie'])


    #import the slicing indices
    slicePath = 'Pickle Objects/VocSlices.p'
    sliceDict = pickle.load(open(slicePath, "rb" )) #contains df of co2 slice indices and matched movie list

    #round the vocs
    voc2015Col = vocRounding(voc2015DfAll)
    voc2013Col = vocRounding(voc2013DfAll)
    voc2013Df = copy.deepcopy(voc2013DfAll)
    voc2015Df = copy.deepcopy(voc2015DfAll)
    voc2013Df.columns = voc2013Col
    voc2015Df.columns = voc2015Col

    #rearrange dataframe to be able to merge them successfully
    voc = voc2015Df.columns[1:]
    voc2015Df = pd.DataFrame(np.transpose(voc2015Df.values)[1:,:], columns =voc2015Df['Time'])
    voc2015Df['voc'] = voc
    voc = index=voc2013Df.columns[1:]
    voc2013Df = pd.DataFrame(np.transpose(voc2013Df.values)[1:,:], columns =voc2013Df['Time'])
    voc2013Df['voc'] = voc

    #join the two voc dataframes (join on the 2013 dataframe)
    vocDf = pd.merge(voc2013Df, voc2015Df, how='inner', on=['voc'])
    #drop voc column
    vocColumn = vocDf['voc']
    vocDf.drop("voc", axis=1, inplace=True)

    #reorientate the vocDf, note need to convert all vocs to float
    vocDf = pd.DataFrame(np.transpose(vocDf.values.astype(float)), columns=vocColumn)
    
    resultsList = list()
    startIndex = 0
    endIndex = 1
    randomisationIterations = 100
    resultsHeader = ['RandomState','VOC','Precision', 'Accuracy']

    for vocIndex in range(startIndex,endIndex):
        for i in range(0,randomisationIterations):
            print('Iteration: ', str(i))

            voc = vocDf.columns[vocIndex]
            print(voc)
            vocData = vocDf[voc]

            print('Process Data')
            #generate normalised screenings
            screeningList, matchedMovies = generateNormalisedScreenings(sliceDict, vocData)

            #movie-based train test split
            #normal screenings
            testScreenings,testMovies,trainScreenings,trainMovies = movieTrainTestSplit(movieList,matchedMovies,screeningList)

            #create input-output df
            testSet = inputOutputDf(testScreenings)
            trainSet = inputOutputDf(trainScreenings)

            #extract labels and features
            featuresTrain = trainSet[:, 0:-1]
            labelsTrain = trainSet[:,-1]
            featuresTest = testSet[:, 0:-1]
            labelsTest = testSet[:,-1]

            print('Run classification')
            #regression
            precisionScore,accuracyScore = ClassificationModel(featuresTrain,labelsTrain, labelsTest,featuresTest)
            resultsList.append([False, voc, precisionScore,accuracyScore])

        print('Write results to file')
        #create results Df
        resultsDf = pd.DataFrame(resultsList,columns=resultsHeader)
        #write df to output file
        resultsPath = str(voc) + 'AutoClassifer_MinMaxScaler.csv'
        resultsDf.to_csv(resultsPath, sep=',', encoding='utf-8')
        print()

main()
