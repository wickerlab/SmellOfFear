import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import cv2
import pickle
import copy

def find_histogram(clt):
  """
  create a histogram with k clusters
  :param: clt
  :return:hist
  """
  numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
  (hist, _) = np.histogram(clt.labels_, bins=numLabels)

  hist = hist.astype("float")
  hist /= hist.sum()

  return hist

def plot_colors2(hist, centroids):
  bar = np.zeros((50, 300, 3), dtype="uint8")
  startX = 0

  for (percent, color) in zip(hist, centroids):
      # plot the relative percentage of each cluster
      endX = startX + (percent * 300)
      cv2.rectangle(bar, (int(startX), 0), (int(endX), 50),
                    color.astype("uint8").tolist(), -1)
      startX = endX

  # return the bar chart
  return bar

#find dominant colour through K-means clustering

def dominantColourOrShadeEvaluation(movieFrames):
    dominantList = list()
    for img in movieFrames:
        clt = KMeans(n_clusters=1)
        clt.fit(img)
        hist = find_histogram(clt)
        bar = plot_colors2(hist, clt.cluster_centers_)
        dominantList.append(list(bar[0][0]))
    return dominantList


def main():

    #import movie list
    movieRuntimesPath = 'Numerical Data//movie_runtimes.csv'
    movieRuntimeDf = pd.read_csv(movieRuntimesPath, usecols = ['movie', 'runtime (mins)', 'effective runtime'], nrows = 9)
    movieList = list(movieRuntimeDf['movie'])

    #note: frame sampling occurred at 1 frame every 3 seconds
    #Read in images and perform kmeans for dominant colour/shade clustering

    movieFrameNumbers = dict()
    movieFrameNumbers['Hobbit 2'] = 3226
    movieFrameNumbers['Buddy'] = 1817
    movieFrameNumbers['Machete Kills'] = 2161
    movieFrameNumbers['Walter Mitty'] = 2292
    movieFrameNumbers['Paranormal Activity'] = 2022
    movieFrameNumbers['The Hunger Games-Catching Fire'] = 2925
    movieFrameNumbers['Star Wars-The Force Awakens'] = 2762

    for movie in movieList:
        movieFrameList = list()
        movieGrayFrameList = list()
        for j in range(1, movieFrameNumbers[movie]+1): #movie frame numbers start at 1 
            number = '{:04d}'.format(j)
            inputPath = 'Features//MovieFrames//' + movie + "//" + movie + "_" + str(number) + '.jpg'
            img = cv2.imread(inputPath)
            img = np.array(img, dtype=np.uint8)
            #convert BGR image to RGB
            img = cv2.cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            #convert RGB to GRAYSCALE
            grayImg = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            #reshape array from (x, y, 3) to (x*y, 3) 
            #3 rows of R G B
            img = img.reshape((img.shape[0] * img.shape[1],3)) 
            #reshape array from (x,y,1) to (x*y)    
            grayImg = grayImg.reshape((grayImg.shape[0] * grayImg.shape[1],1)) 
            #append to array
            movieFrameList.append(img)
            movieGrayFrameList.append(grayImg)
        
        
        print(movie + ' dominant colour processing')
        #movie frames both grey and colour loaded
        dominantColourList = dominantColourOrShadeEvaluation(movieFrameList)
        print(movie + ' dominant shade processing')
        dominantShadeList = dominantColourOrShadeEvaluation(movieGrayFrameList)
        
        print(movie + ' save pickle object')
        dominantColourFilename = 'Pickle Objects//' + movie + 'DominantColour.p'
        dominantShadeFilename = 'Pickle Objects//' + movie + 'DominantShade.p' 
        pickle.dump(dominantColourList, open(dominantColourFilename, 'wb'))
        pickle.dump(dominantShadeList, open(dominantShadeFilename, 'wb'))
    

main()

