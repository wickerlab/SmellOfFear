import pandas as pd
import numpy as np
import scipy.signal as ss
import math
import datetime 
import pickle
import random
import copy

def deltaVOCCalculation(normalisedScreenings, voc):
    deltaNormalisedScreenings = list()
    for screening in normalisedScreenings:
        deltaDf = pd.DataFrame(columns=[list(screening.columns)[1]])
        for rowIndex in range(0, len(screening)-1):
            delta = screening[voc].values[rowIndex+1]-screening[voc].values[rowIndex]
            deltaDf.loc[rowIndex] = delta
        deltaNormalisedScreenings.append(deltaDf)
    return deltaNormalisedScreenings        

def windowing(screening, vocTimeList, co2Df):
    
    #add interval to start and end of window
    startTime = list(screening['Time'])[0]
    startTimeIndex = vocTimeList.index(startTime) - 5
    endTimeIndex = startTimeIndex + len(screening) + 9
    windowDf = co2Df.loc[startTimeIndex:endTimeIndex][:] #adjust the voc screening 
    windowDf = normalisation([windowDf])
    #create a dataframe of the windows
    windowedList = list()
    for i in range(0, len(screening)):
        window = windowDf[0].loc[startTimeIndex:startTimeIndex+9][:]
        windowedList.append(window)
        startTimeIndex = startTimeIndex + 1
    return windowedList

def preliminaryAlignment(runtime, vocTime, vocTimeList,preliminaryAlignmentTolerance,co2Df):
    effectiveRuntime = (runtime + preliminaryAlignmentTolerance) * 2 #tolerance added is 15mins and then multiplied by 2 to get the number of 30s intervals
    vocIndex = vocTimeList.index(vocTime)
    vocEndIndex = vocIndex + effectiveRuntime
    vocWindow = co2Df.loc[vocIndex:vocEndIndex][:]
    return vocWindow

def calculateDeltaBetweenPeaks(vocWindow):

    #find peaks 
    peakList = list()
    normalisedPeakList = list()
    #find_peaks returns the index values of the peaks within the VOC frame 
    peaks = ss.find_peaks(vocWindow[:]['CO2'].values)
    
    #Using the index values find the actual values of the peaks 
    deltaList = list()
    if len(peaks[0]) != 0:
        for peakIndex in peaks[0]:
            peakList.append(vocWindow[:]['CO2'].values[peakIndex])

        #normalise the peaks (divide by highest VOC value)
        maxPeak = max(peakList)
        for peakValue in peakList:
            normalisedPeakList.append(peakValue/maxPeak) 

        #calculate the gradient and distance between peaks
        #the gradientList and distanceList for vocFrame
        
        for peakIndex in range(1, len(normalisedPeakList)):
            prevPeak = normalisedPeakList[peakIndex-1]
            currPeak = normalisedPeakList[peakIndex]
            delta = currPeak - prevPeak
            deltaList.append(delta)
 
    return deltaList, peakList



def gradientAlignment(vocWindow,gradThreshold, effectiveRuntime,vocTime,movieMatched):

    vocList = list()
    
    deltaList, peakList = calculateDeltaBetweenPeaks(vocWindow)

    if len(deltaList) != 0:

        #apply constraints to trim the voc window
        frontIndex = round(len(deltaList)*0.8) #only check the last 20% of the voc window
        vocConstraintWindow = deltaList[frontIndex:]

        if min(vocConstraintWindow) > gradThreshold:

            #if the min gradient in the frame is larger than the threshold then just cut off the last peak
            lastPeakIndex = list(vocWindow[:]['CO2'].values).index(peakList[-1])
            firstIndex = lastPeakIndex - effectiveRuntime 
            vocWindow = vocWindow[firstIndex:lastPeakIndex][:]
            vocList.append(vocWindow)
            
        else: 

            #if min gradient in frame is less than threshold then cut off the peak that starts that gradient
            #find the first grad that is lower than the threshold

            for grad in vocConstraintWindow:
                if grad < gradThreshold:
                    gradIndex = deltaList.index(grad)
                    associatedPeak = peakList[gradIndex]
                    endIndex = list(vocWindow[:]['CO2'].values).index(associatedPeak)

                    firstIndex = endIndex - effectiveRuntime

                    if firstIndex > 0: #positive index
                        vocWindow = vocWindow[firstIndex:endIndex][:]
                        vocList.append(vocWindow)
                        break

    return vocList

def dataAlignment(scheduledTimeList, movieScreeningList, movieList, filledPercentageList, vocTimeList, matchedMovieList, timeList, vocScreenings, co2Df, preliminaryAlignmentTolerance, gradThreshold, filledPercentageConstraint,movieRuntimeDf, originalVOCFrames):
   
    for vocTime in vocTimeList: 
    #match timing with movie scheduled timing 
        try:
            timeIndex = scheduledTimeList.index(vocTime)
        except:
            continue 
            
        if vocTime not in timeList:
            if filledPercentageList[timeIndex] > filledPercentageConstraint: #only use well filled movies
                try:
                    movieMatched = movieScreeningList[timeIndex]    
                    movieIndex = movieList.index(movieMatched)      
                except:
                    continue 
                    
                effectiveRuntime = movieRuntimeDf.loc[movieIndex]['effective runtime']
                runtime = movieRuntimeDf.loc[movieIndex]['runtime (mins)']
                vocWindow = preliminaryAlignment(runtime, vocTime, vocTimeList, preliminaryAlignmentTolerance, co2Df)
                vocList = gradientAlignment(vocWindow,gradThreshold, effectiveRuntime,vocTime,movieMatched)
                if len(vocList) > 0:
                    originalVOCFrames.append(vocWindow)
                    timeList.append(vocTime)
                    matchedMovieList.append(movieMatched)
                    vocScreenings = vocScreenings + vocList
                
    
    return vocScreenings, matchedMovieList, timeList, originalVOCFrames



def normalisation(vocScreenings):
    normalisedVOCList = list()
    for screening in vocScreenings:
        normalisedVOCFrame = screening[:]['CO2'].values/max(screening[:]['CO2'].values)
        normalisedScreening = screening[:][:]
        normalisedScreening[:]['CO2'] = normalisedVOCFrame
        normalisedVOCList.append(normalisedScreening)
    return normalisedVOCList

def errorAdjustment(vocList, timeList, matchedMovieList,originalVOCFrames,movieRuntimeDf):
    #VOC Screenings to be manually editted to after inspection

    #The Hunger Games: Catching Fire 27-12-2013 13:15
    #Buddy 29-12-2013 19:30
    #Walter Mitty 02-01-2014 17:15
    #The Hunger Games: Catching Fire 05-01-2014 13:45
    #Walter Mitty 05-01-2014 17:15
    #The Hunger Games: Catching Fire 07-01-2014 13:45
    #Paranormal Activity 09-01-2014 20:35
    #Hobbit 2 10-01-2014 16:30
    #Paranormal Activity 10-01-2014 22:35
    #Help I Shrunk 27-12-2015 11:30
    #Help I Shrunk 30-12-2015 11:30
    #Help I Shrunk 02-01-2016 11:30
    # Help I Shrunk 03-01-2016 11:30
    #I'm Off Then 27-12-2015 20:00
    #I'm Off Then 30-12-2015 20:00
    #I'm Off Then 31-12-2015 20:00
    #I'm Off Then 02-01-2016 17:30
    #I'm Off Then 02-01-2016 20:00
    #I'm Off Then 03-01-2016 17:30
    #Star Wars-A Force Awakens 22-12-2015 22:30
    #Star Wars-A Force Awakens 28-12-2015 22:30
    #Star Wars-A Force Awakens 29-12-2015 22:30

    errorList = ['27-12-2013 13:15', '29-12-2013 19:30', '02-01-2014 17:15', '05-01-2014 13:45',
                '05-01-2014 17:15', '07-01-2014 13:45', '09-01-2014 20:35', '10-01-2014 16:30', 
                '10-01-2014 22:35', '22-12-2015 22:30', '28-12-2015 22:30', '29-12-2015 22:30', 
                '27-12-2015 20:00', '30-12-2015 20:00', '31-12-2015 20:00', '02-01-2016 17:30',
                '02-01-2016 20:00','03-01-2016 17:30', '30-12-2015 11:30', 
                '02-01-2016 11:30', '03-01-2016 11:30']
                
    adjustedVOCList = copy.deepcopy(vocList)

    movieList = list(movieRuntimeDf['movie'])

    for errorDate in errorList:
        
        errorIndex = timeList.index(errorDate)
        matchedMovie = matchedMovieList[errorIndex]
        movieIndex = movieList.index(matchedMovie)
        vocFrame = originalVOCFrames[errorIndex]
   
        #Star Wars
        if errorDate == '22-12-2015 22:30':
            endOfMovie = '23-12-2015 00:55'
            movieIndex = movieList.index('Star Wars-The Force Awakens')
            effectiveRuntime = movieRuntimeDf.loc[movieIndex]['effective runtime']
            vocEndIndex = list(vocFrame[:]['Time'].values).index(endOfMovie)
            vocStartIndex = vocEndIndex - effectiveRuntime
            vocWindow = vocFrame[vocStartIndex:vocEndIndex][:]
            adjustedVOCList[errorIndex] = vocWindow
        elif errorDate == '28-12-2015 22:30':
            endOfMovie = '29-12-2015 00:59'
            movieIndex = movieList.index('Star Wars-The Force Awakens')
            effectiveRuntime = movieRuntimeDf.loc[movieIndex]['effective runtime']
            vocEndIndex = list(vocFrame[:]['Time'].values).index(endOfMovie)
            vocStartIndex = vocEndIndex - effectiveRuntime
            vocWindow = vocFrame[vocStartIndex:vocEndIndex][:]
            adjustedVOCList[errorIndex] = vocWindow
        elif errorDate == '29-12-2015 22:30':
            endOfMovie = '30-12-2015 01:15'
            movieIndex = movieList.index('Star Wars-The Force Awakens')
            effectiveRuntime = movieRuntimeDf.loc[movieIndex]['effective runtime']
            vocEndIndex = list(vocFrame[:]['Time'].values).index(endOfMovie)
            vocStartIndex = vocEndIndex - effectiveRuntime
            vocWindow = vocFrame[vocStartIndex:vocEndIndex][:]
            adjustedVOCList[errorIndex] = vocWindow

        #2013 movies
        elif errorDate == '27-12-2013 13:15':
            endOfMovie = '27-12-2013 15:59' 
            movieIndex = movieList.index(matchedMovie)
            effectiveRuntime = movieRuntimeDf.loc[movieIndex]['effective runtime']
            vocEndIndex = list(vocFrame[:]['Time'].values).index(endOfMovie)
            vocStartIndex = vocEndIndex - effectiveRuntime
            vocWindow = vocFrame[vocStartIndex:vocEndIndex][:]
            adjustedVOCList[errorIndex] = vocWindow
        elif errorDate == '29-12-2013 19:30':
            endOfMovie = '29-12-2013 21:28' 
            movieIndex = movieList.index(matchedMovie)
            effectiveRuntime = movieRuntimeDf.loc[movieIndex]['effective runtime']
            vocEndIndex = list(vocFrame[:]['Time'].values).index(endOfMovie)
            vocStartIndex = vocEndIndex - effectiveRuntime
            vocWindow = vocFrame[vocStartIndex:vocEndIndex][:]
            adjustedVOCList[errorIndex] = vocWindow  
        elif errorDate == '02-01-2014 17:15':
            endOfMovie = '02-01-2014 19:21' 
            movieIndex = movieList.index(matchedMovie)
            effectiveRuntime = movieRuntimeDf.loc[movieIndex]['effective runtime']
            vocEndIndex = list(vocFrame[:]['Time'].values).index(endOfMovie)
            vocStartIndex = vocEndIndex - effectiveRuntime
            vocWindow = vocFrame[vocStartIndex:vocEndIndex][:]
            adjustedVOCList[errorIndex] = vocWindow  
        elif errorDate == '05-01-2014 13:45':
            endOfMovie = '05-01-2014 16:21' 
            movieIndex = movieList.index(matchedMovie)
            effectiveRuntime = movieRuntimeDf.loc[movieIndex]['effective runtime']
            vocEndIndex = list(vocFrame[:]['Time'].values).index(endOfMovie)
            vocStartIndex = vocEndIndex - effectiveRuntime
            vocWindow = vocFrame[vocStartIndex:vocEndIndex][:]
            adjustedVOCList[errorIndex] = vocWindow        
        elif errorDate == '05-01-2014 17:15':
            endOfMovie = '05-01-2014 19:21' 
            movieIndex = movieList.index(matchedMovie)
            effectiveRuntime = movieRuntimeDf.loc[movieIndex]['effective runtime']
            vocEndIndex = list(vocFrame[:]['Time'].values).index(endOfMovie)
            vocStartIndex = vocEndIndex - effectiveRuntime
            vocWindow = vocFrame[vocStartIndex:vocEndIndex][:]
            adjustedVOCList[errorIndex] = vocWindow  
        elif errorDate == '07-01-2014 13:45':
            endOfMovie = '07-01-2014 16:10' 
            movieIndex = movieList.index(matchedMovie)
            effectiveRuntime = movieRuntimeDf.loc[movieIndex]['effective runtime']
            vocEndIndex = list(vocFrame[:]['Time'].values).index(endOfMovie)
            vocStartIndex = vocEndIndex - effectiveRuntime 
            vocWindow = vocFrame[vocStartIndex:vocEndIndex][:]
            adjustedVOCList[errorIndex] = vocWindow              
        elif errorDate == '09-01-2014 20:35':
            endOfMovie = '09-01-2014 22:28'
            movieIndex = movieList.index(matchedMovie)
            effectiveRuntime = movieRuntimeDf.loc[movieIndex]['effective runtime']
            vocEndIndex = list(vocFrame[:]['Time'].values).index(endOfMovie)
            vocStartIndex = vocEndIndex - effectiveRuntime
            vocWindow = vocFrame[vocStartIndex:vocEndIndex][:]
            adjustedVOCList[errorIndex] = vocWindow  
        elif errorDate == '10-01-2014 16:30':
            endOfMovie = '10-01-2014 19:50'
            movieIndex = movieList.index(matchedMovie)
            effectiveRuntime = movieRuntimeDf.loc[movieIndex]['effective runtime']
            vocEndIndex = list(vocFrame[:]['Time'].values).index(endOfMovie)
            vocStartIndex = vocEndIndex - effectiveRuntime
            vocWindow = vocFrame[vocStartIndex:vocEndIndex][:]
            adjustedVOCList[errorIndex] = vocWindow              
        elif errorDate == '10-01-2014 22:35':
            endOfMovie = '11-01-2014 00:25'
            movieIndex = movieList.index(matchedMovie)
            effectiveRuntime = movieRuntimeDf.loc[movieIndex]['effective runtime']
            vocEndIndex = list(vocFrame[:]['Time'].values).index(endOfMovie)
            vocStartIndex = vocEndIndex - effectiveRuntime
            vocWindow = vocFrame[vocStartIndex:vocEndIndex][:]
            adjustedVOCList[errorIndex] = vocWindow  

        #I'm Off Then
        elif errorDate == '27-12-2015 20:00':
            endOfMovie = '27-12-2015 21:54'
            movieIndex = movieList.index('I\'m Off Then')
            effectiveRuntime = movieRuntimeDf.loc[movieIndex]['effective runtime']
            vocEndIndex = list(vocFrame[:]['Time'].values).index(endOfMovie)
            vocStartIndex = vocEndIndex - effectiveRuntime
            vocWindow = vocFrame[vocStartIndex:vocEndIndex][:]
            adjustedVOCList[errorIndex] = vocWindow            
        elif errorDate == '30-12-2015 20:00':
            endOfMovie = '30-12-2015 21:52'
            movieIndex = movieList.index('I\'m Off Then')
            effectiveRuntime = movieRuntimeDf.loc[movieIndex]['effective runtime']
            vocEndIndex = list(vocFrame[:]['Time'].values).index(endOfMovie)
            vocStartIndex = vocEndIndex - effectiveRuntime
            vocWindow = vocFrame[vocStartIndex:vocEndIndex][:]
            adjustedVOCList[errorIndex] = vocWindow            
        elif errorDate == '31-12-2015 20:00':
            endOfMovie = '31-12-2015 21:53'
            movieIndex = movieList.index('I\'m Off Then')
            effectiveRuntime = movieRuntimeDf.loc[movieIndex]['effective runtime']
            vocEndIndex = list(vocFrame[:]['Time'].values).index(endOfMovie)
            vocStartIndex = vocEndIndex - effectiveRuntime
            vocWindow = vocFrame[vocStartIndex:vocEndIndex][:]
            adjustedVOCList[errorIndex] = vocWindow            
        elif errorDate == '02-01-2016 17:30':
            endOfMovie = '02-01-2016 19:22'
            movieIndex = movieList.index('I\'m Off Then')
            effectiveRuntime = movieRuntimeDf.loc[movieIndex]['effective runtime']
            vocEndIndex = list(vocFrame[:]['Time'].values).index(endOfMovie)
            vocStartIndex = vocEndIndex - effectiveRuntime
            vocWindow = vocFrame[vocStartIndex:vocEndIndex][:]
            adjustedVOCList[errorIndex] = vocWindow  
        elif errorDate == '02-01-2016 20:00':
            endOfMovie = '02-01-2016 21:53'
            movieIndex = movieList.index('I\'m Off Then')
            effectiveRuntime = movieRuntimeDf.loc[movieIndex]['effective runtime']
            vocEndIndex = list(vocFrame[:]['Time'].values).index(endOfMovie)
            vocStartIndex = vocEndIndex - effectiveRuntime
            vocWindow = vocFrame[vocStartIndex:vocEndIndex][:]
            adjustedVOCList[errorIndex] = vocWindow
        elif errorDate == '03-01-2016 17:30':
            endOfMovie = '03-01-2016 19:17'
            movieIndex = movieList.index('I\'m Off Then')
            effectiveRuntime = movieRuntimeDf.loc[movieIndex]['effective runtime']
            vocEndIndex = list(vocFrame[:]['Time'].values).index(endOfMovie)
            vocStartIndex = vocEndIndex - effectiveRuntime
            vocWindow = vocFrame[vocStartIndex:vocEndIndex][:]
            adjustedVOCList[errorIndex] = vocWindow
        
        #Help I shrunk the teacher   
        elif errorDate == '30-12-2015 11:30':
            endOfMovie = '30-12-2015 13:26'
            movieIndex = movieList.index('Help, I Shrunk My Teacher')
            effectiveRuntime = movieRuntimeDf.loc[movieIndex]['effective runtime']
            vocEndIndex = list(vocFrame[:]['Time'].values).index(endOfMovie)
            vocStartIndex = vocEndIndex - effectiveRuntime
            vocWindow = vocFrame[vocStartIndex:vocEndIndex][:]
            adjustedVOCList[errorIndex] = vocWindow
        elif errorDate == '02-01-2016 11:30':
            endOfMovie = '02-01-2016 13:23'
            movieIndex = movieList.index('Help, I Shrunk My Teacher')
            effectiveRuntime = movieRuntimeDf.loc[movieIndex]['effective runtime']
            vocEndIndex = list(vocFrame[:]['Time'].values).index(endOfMovie)
            vocStartIndex = vocEndIndex - effectiveRuntime + 1
            vocWindow = vocFrame[vocStartIndex:vocEndIndex][:]
            adjustedVOCList[errorIndex] = vocWindow           
        elif errorDate == '03-01-2016 11:30':
            endOfMovie = '03-01-2016 13:23'
            movieIndex = movieList.index('Help, I Shrunk My Teacher')
            effectiveRuntime = movieRuntimeDf.loc[movieIndex]['effective runtime']
            vocEndIndex = list(vocFrame[:]['Time'].values).index(endOfMovie)
            vocStartIndex = vocEndIndex - effectiveRuntime
            vocWindow = vocFrame[vocStartIndex:vocEndIndex][:]
            adjustedVOCList[errorIndex] = vocWindow

    return adjustedVOCList

            


def main():
    #user macros
    gradThreshold = -0.045
    preliminaryAlignmentTolerance = 50
    filledPercentageConstraint = 10 #movie must have atleast 10% filled to get a decent reading
    voc = 'CO2'

    #read in the various csvs
    #2013 Dataset
    vocPath = 'Numerical Data/CO2data.csv'
    co2Df = pd.read_csv(vocPath, usecols = ['Time',voc], header = 0, nrows = 74208)
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
    vocPath = 'Numerical Data/final_data_ppb.csv'
    cinestar2015Co2Df = pd.read_csv(vocPath, usecols = ['Time', voc])

    #Standardize times within the VOC dataset
    #VOC timings with datetime object
    for i in range(0,co2Df.shape[0]):
        vocTime = co2Df.loc[i]['Time']
        vocTime = vocTime[1:len(vocTime)-1]
        date = datetime.datetime.strptime(vocTime, "%m/%d/%Y %H:%M:%S")
        co2Df.at[i,'Time'] = date.strftime('%d-%m-%Y %H:%M')
    for i in range(0, cinestar2015Co2Df.shape[0]):
        vocTime = cinestar2015Co2Df.loc[i]['Time']
        date = datetime.datetime.strptime(vocTime, "%d/%m/%Y %H:%M")
        cinestar2015Co2Df.at[i,'Time'] = date.strftime('%d-%m-%Y %H:%M')

    #Standardize times within the cinema movie schedule
    #2013
    for i in range(0,movingScreeningsDf.shape[0]):
        scheduledTime = movingScreeningsDf.loc[i]['scheduled']
        scheduledTimeObj = datetime.datetime.strptime(scheduledTime, "%d/%m/%Y %H:%M")
        scheduledTime = scheduledTimeObj.strftime('%d-%m-%Y %H:%M')
        movingScreeningsDf.at[i,'scheduled'] = scheduledTime
    #2015 Star Wars
    for i in range(0, starWarsScreeningDf.shape[0]):
        beginTime = starWarsScreeningDf.loc[i]['Start']
        beginTimeObj = datetime.datetime.strptime(beginTime,  "%d/%m/%Y %H:%M")
        beginTime = beginTimeObj.strftime('%d-%m-%Y %H:%M')
        starWarsScreeningDf.at[i,'Start'] = beginTime
    #2015 I'm Off Then
    for i in range(0, imOffThenScreeningDf.shape[0]):
        beginTime = imOffThenScreeningDf.loc[i]['Start']
        beginTimeObj = datetime.datetime.strptime(beginTime,  "%d/%m/%Y %H:%M")
        beginTime = beginTimeObj.strftime('%d-%m-%Y %H:%M')
        imOffThenScreeningDf.at[i,'Start'] = beginTime    
    #2015 Help, I Shrunk The Teacher
    for i in range(0, helpIShrunkTheTeacherScreeningDf.shape[0]):
        beginTime = helpIShrunkTheTeacherScreeningDf.loc[i]['Start']
        beginTimeObj = datetime.datetime.strptime(beginTime,  "%d/%m/%Y %H:%M")
        beginTime = beginTimeObj.strftime('%d-%m-%Y %H:%M')
        helpIShrunkTheTeacherScreeningDf.at[i,'Start'] = beginTime

    #Data Alignment 
    #2013 Alignment
    scheduledTimeList = list(movingScreeningsDf.loc[:]['scheduled'])
    movieScreeningList = list(movingScreeningsDf.loc[:]['movie'])
    movieList = list(movieRuntimeDf.loc[:]['movie'])
    filledPercentageList = list(movingScreeningsDf.loc[:]['filled %'])
    vocTimeList = list(co2Df.loc[:]['Time'])

    matchedMovieList = list()
    timeList = list() 
    vocScreenings = list()
    originalVOCFrames = list()
    vocScreenings, matchedMovieList, timeList, originalVOCFrames = dataAlignment(scheduledTimeList, movieScreeningList, movieList, filledPercentageList, vocTimeList, matchedMovieList, timeList, vocScreenings, co2Df, preliminaryAlignmentTolerance, gradThreshold,filledPercentageConstraint,movieRuntimeDf,originalVOCFrames)

    #2015 Star Wars
    scheduledTimeList = list(starWarsScreeningDf.loc[:]['Start'])
    vocTimeList = list(cinestar2015Co2Df.loc[:]['Time'])
    filledPercentageList = list(starWarsScreeningDf.loc[:]['filled %'])
    movieScreeningList = list(starWarsScreeningDf.loc[:]['Film'])
    vocScreenings, matchedMovieList, timeList, originalVOCFrames = dataAlignment(scheduledTimeList, movieScreeningList, movieList, filledPercentageList, vocTimeList, matchedMovieList, timeList, vocScreenings, cinestar2015Co2Df, preliminaryAlignmentTolerance, gradThreshold,filledPercentageConstraint,movieRuntimeDf,originalVOCFrames)

    #2015 I'm Off Then
    scheduledTimeList = list(imOffThenScreeningDf.loc[:]['Start'])
    filledPercentageList = list(imOffThenScreeningDf.loc[:]['filled %'])
    movieScreeningList = list(imOffThenScreeningDf.loc[:]['Film'])   
    vocScreenings, matchedMovieList, timeList, originalVOCFrames = dataAlignment(scheduledTimeList, movieScreeningList, movieList, filledPercentageList, vocTimeList, matchedMovieList, timeList, vocScreenings, cinestar2015Co2Df, preliminaryAlignmentTolerance, gradThreshold,filledPercentageConstraint,movieRuntimeDf,originalVOCFrames)

    #2015 Help, I Shrunk the Teacher
    scheduledTimeList = list(helpIShrunkTheTeacherScreeningDf.loc[:]['Start'])
    filledPercentageList = list(helpIShrunkTheTeacherScreeningDf.loc[:]['filled %'])
    movieScreeningList = list(helpIShrunkTheTeacherScreeningDf.loc[:]['Film'])     
    vocScreenings, matchedMovieList, timeList, originalVOCFrames = dataAlignment(scheduledTimeList, movieScreeningList, movieList, filledPercentageList, vocTimeList, matchedMovieList, timeList, vocScreenings, cinestar2015Co2Df, preliminaryAlignmentTolerance, gradThreshold,filledPercentageConstraint,movieRuntimeDf,originalVOCFrames)

    #error adjustment
    adjustedScreenings = errorAdjustment(vocScreenings, timeList, matchedMovieList,originalVOCFrames,movieRuntimeDf)

    #normalised vocs
    normalisedScreenings =  normalisation(adjustedScreenings)
    
    #save the normalised vocs
    #compile the matchedMovie, timeList and the screenings into a dictionary then save them
    if not(isWindowing):
        normalisedScreeningsDict = {'screenings':normalisedScreenings, 'matchedMovies':matchedMovieList}
        pickle.dump(normalisedScreeningsDict, open( "normalisedScreeningsDict.p", "wb" ) ) 

    #applying windowing
    if isWindowing:
        windowedNormalisedScreenings = list()
        for screening in normalisedScreenings:
            #find year of movie and then to figure out what VOC dataset to give it
            year = list(screening['Time'])[0][6:10]
            if year == '2013' or year == '2014':
                vocTimeList =list(co2Df['Time'])
                windowList = windowing(screening, vocTimeList, co2Df)
            elif year == '2015' or year == '2016':
                vocTimeList = list(cinestar2015Co2Df['Time'])
                windowList = windowing(screening, vocTimeList, cinestar2015Co2Df)
            windowedNormalisedScreenings.append(windowList)

        normalisedWindowedScreeningsDict = {'screenings':windowedNormalisedScreenings, 'matchedMovies':matchedMovieList, 'timeList':timeList}
        pickle.dump(normalisedWindowedScreeningsDict, open( "normalisedWindowedScreeningsDict.p", "wb" ) ) 
    
    #calculate delta voc dataset
    if isDelta:
        deltaScreenings = deltaVOCCalculation(normalisedScreenings, voc)
        deltaScreeningsDict = {'screenings':deltaScreenings, 'matchedMovies':matchedMovieList,'timeList':timeList}
        pickle.dump(deltaScreeningsDict, open( "deltaScreeningsDict.p", "wb" ) )


main()