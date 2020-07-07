import numpy as np
import pandas as pd
import librosa
import pickle


#import csv data files
movieRuntimePath = 'Numerical Data//movie_runtimes.csv'
movieRuntimeDf = pd.read_csv(movieRuntimePath, usecols = ['movie', 'runtime (mins)', 'effective runtime'], header = 0)
#create a list of movies
movieList = list(movieRuntimeDf['movie'])

movieList = [movieList[-1]] 

sr = 22050 #sampling rate

for movie in movieList:

    index = movieList.index(movie)
    #load audio
    basePath = '/home/sof/Notebooks/Pickle Objects/Raw Audio File Pickle Objects/' #enter path to audio pickle objects
    moviePath = basePath + movie + '.p'
    try:
        y = pickle.load(open(moviePath,"rb")) 
        print('FOUND: ', movie)
    except FileNotFoundError:
        #movie files that we do not have
        print(movie)
        continue

    #split the audio into 30s intervals
    runtime = movieRuntimeDf.loc[index]['runtime (mins)'] 
    intervals = runtime * 2
    x = np.array_split(y,intervals)

    featureDict = dict()
    logMelList = list()
    chromaList = list()
    tempoList = list()
    mfccList = list()
    specCentroidList = list()
    specContrastList = list()
    tonnetzList = list()

    for k in x:

        #mel power spectrogram
        mel = librosa.feature.melspectrogram(y=k,sr=sr)
        #convert to log scale (dB) and use peak power as a reference
        logMel = librosa.power_to_db(mel, ref=np.max)
        #logMel - take mean of every column
        logMel = np.mean(logMel, axis = 1)

        #chroma - pitch class information
        chroma = librosa.feature.chroma_cqt(y = k, sr=sr)
        #chroma - take mean of every column
        chroma = np.mean(chroma, axis = 1)

        #estimated tempo information
        tempo, beat_frames = librosa.beat.beat_track(y = k,sr=sr)

        #mfcc 
        mfcc = librosa.feature.mfcc(y=k, sr=sr, n_mfcc = 40) #40 is the amount of cepstral vectors 
        #mfcc - take mean of every column
        mfcc = np.mean(mfcc, axis = 1)        

        #spectral centroid - relates to brightness of sound
        specCentroid = librosa.feature.spectral_centroid(y = k, sr=sr)
        specCentroid = np.mean(specCentroid, axis = 1)      

        #spectral contrast
        specContrast = librosa.feature.spectral_contrast(y = k, sr=sr)
        specContrast = np.mean(specContrast, axis = 1)   

        #tonnetz - tonal centroid features
        tonnetz = librosa.feature.tonnetz(y = k, sr = sr)
        tonnetz = np.mean(tonnetz, axis = 1)  

        logMelList.append(logMel)
        chromaList.append(chroma)
        tempoList.append(tempo)
        mfccList.append(mfcc)
        specCentroidList.append(specCentroid)
        specContrastList.append(specContrast)
        tonnetzList.append(tonnetz)
    
    featureDict['logMel'] = logMelList
    featureDict['chroma'] = chromaList
    featureDict['tempo'] = tempoList
    featureDict['mfcc'] = mfccList
    featureDict['specCentroid'] = specCentroidList
    featureDict['specContrast'] = specContrastList
    featureDict['tonnetz'] = tonnetzList
    
    print(movie + ' finished processing')

    audioFeaturePath = 'Pickle Objects//Audio Feature Pickle Objects//' + movie + '.p'
    pickle.dump(featureDict, open(audioFeaturePath, "wb" ))
    
    print(movie + ' saved as pickle object')




