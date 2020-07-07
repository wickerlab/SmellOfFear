"""
Run NLP Server: Navigate to the directory that contains StanfordNLP then run the following code on Terminal
    java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -annotators "sentiment" -port 9000 -timeout 30000
"""

import pandas as pd
import numpy as np
import pickle
import copy
from pycorenlp import StanfordCoreNLP
import logging
import json
from datetime import time
import copy

def divideSubsIntoSegments(lines, runtime):
    
    refEndSec = 30
    refEndMin = 0
    refEndHour = 0
    referenceEndTime = time(refEndHour, refEndMin, refEndSec)
    refStartSec = 0
    refStartMin = 0
    refStartHour = 0
    referenceStartTime = time(refStartHour, refStartMin, refStartSec)
    startTime = None
    
    subtitleIntervals = list()
    
    for segment in range(0, runtime):
        #parse of timing information
        subtitleSegment = list()
        for rowIndex in range(0,len(lines)):
            line = str(lines[rowIndex])
            if len(line) > 15:
                arrow = line[13] + line[14] + line[15]
                if arrow == '-->':
                    #timing information detected
                    #parse the actual time
                    startTime = line[0:8] #extract the start time
                    startHour = int(startTime[0] + startTime[1])
                    startMinutes = int(startTime[3] + startTime[4])
                    startSeconds = int(startTime[6] + startTime[7])
                    startTime = time(startHour,startMinutes,startSeconds)
                    
                    if startTime > referenceEndTime:
                        
                        subtitleIntervals.append(subtitleSegment)
                        
                        refStartSec = refStartSec + 30
                        if refStartSec == 60:
                            refStartSec = 0
                            refStartMin = refStartMin + 1
                        if refStartMin == 60:
                            refStartMin = 0
                            refStartHour = refStartHour + 1
                        
                        refEndSec = refEndSec + 30
                        if refEndSec == 60:
                            refEndMin = refEndMin + 1
                            refEndSec = 0
                        if refEndMin == 60:
                            refEndHour = refEndHour + 1
                            refEndMin = 0
                            
                        referenceEndTime = time(refEndHour, refEndMin, refEndSec)
                        referenceStartTime = time(refStartHour, refStartMin, refStartSec)
                        
                        break
                        
                    continue
                    
            if startTime != None:
                if startTime >= referenceStartTime and startTime <= referenceEndTime:
                    subtitleSegment.append(line)   
                    
    #append final segment to interval list
    subtitleIntervals.append(subtitleSegment)
    return subtitleIntervals 

#remove any uncessary lines and unecessary characters within dialog lines
#remove any uncessary lines and unecessary characters within dialog lines
def editSubtitleData(subtitleIntervals):
    
    parsedSubtitleIntervals = list()
    htmlFlag = False
    
    for index in range(0,len(subtitleIntervals)):
        subtitleSegment = subtitleIntervals[index]
        modifiedSegment = str()
        if len(subtitleSegment) != 0:
            #parse the segment for any uncessary characters line by line
            for rowIndex in range(0, len(subtitleSegment)):
                line = subtitleSegment[rowIndex]
                parsedLine = str()
                for char in line:
                    #if the character is not a digit then continue to process
                    if not(char.isdigit()) and char != '\n':
                        #remove all html elements e.g. <i>, <b>
                        if char == '<': 
                            htmlFlag = True
                        if char == '>':
                            htmlFlag = False
                        if not(htmlFlag) and char != '>' and char != '\'':
                            parsedLine = parsedLine + char
                if len(parsedLine) != 0:
                    modifiedSegment = modifiedSegment + ' ' + parsedLine      
            parsedSubtitleIntervals.append(modifiedSegment.strip())
        else:
            parsedSubtitleIntervals.append(subtitleSegment)
        
    return parsedSubtitleIntervals



def main():
    movieRuntimePath = 'Numerical Data//movie_runtimes.csv'
    movieRuntimeDf = pd.read_csv(movieRuntimePath, usecols = ['movie', 'effective runtime'])
    movieList = list(movieRuntimeDf['movie'])

    nlp = StanfordCoreNLP('http://localhost:9000')

    for movie in movieList:

        if movie != 'Buddy':
            subPath = 'Features//Subtitles SRT//' + movie + '.srt'
            subs = open(subPath, mode = 'r', encoding='utf-8-sig')
            subs = subs.readlines() #contains each line within the document
            movieIndex = movieList.index(movie)
            segmentList = divideSubsIntoSegments(subs, movieRuntimeDf['effective runtime'][movieIndex])
            subtitleList = editSubtitleData(segmentList)
        else:
            #because buddy was in german it had to be translated first
            buddySubtitlePath = 'Pickle Objects/buddyEngTranslated.p'
            subtitleList = pickle.load(open(buddySubtitlePath, "rb" ))
            movieIndex = movieList.index(movie)
        
        #movie padding or movie removal
        if movieRuntimeDf['effective runtime'][movieIndex] != len(subtitleList):
            if movieRuntimeDf['effective runtime'][movieIndex] < len(subtitleList):
                subtitleList = subtitleList[:movieRuntimeDf['effective runtime'][movieIndex]]
            else:
                while movieRuntimeDf['effective runtime'][movieIndex] != len(subtitleList):
                    subtitleList.append([])


        #For movies with english subtitle scripts
        print(movie + ' begin processing')
        sentimentMovie = list()
        for segment in subtitleList:
            if len(segment) != 0:
                sentimentSegment = dict()
                res = nlp.annotate(segment, properties={'annotators': 'sentiment','outputFormat': 'json','timeout': 30000})
                sentimentSegment['sentiment'] = res['sentences'][0]['sentiment']
                sentimentSegment['sentimentValue'] = res['sentences'][0]['sentimentValue']
                sentimentMovie.append(copy.deepcopy(sentimentSegment))
            else:
                sentimentMovie.append([])
        
        print(movie + ' saving movie as pickle object')
        sentimentPath ='Pickle Objects//' + movie + '.p'
        pickle.dump(sentimentMovie, open(sentimentPath, 'wb'))
        
main()
        
            