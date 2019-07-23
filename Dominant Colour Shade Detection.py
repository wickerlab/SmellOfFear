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

def dominantColourOrShadeEvaluation(img):

    clt = KMeans(n_clusters=1)
    clt.fit(img)
    hist = find_histogram(clt)
    bar = plot_colors2(hist, clt.cluster_centers_)
    dominantObject = list(bar[0][0])
    return dominantObject


def main():

    #import movie list
    movieRuntimesPath = 'Numerical Data//movie_runtimes.csv'
    movieRuntimeDf = pd.read_csv(movieRuntimesPath, usecols = ['movie', 'runtime (mins)', 'effective runtime'], header=0)
    movieList = list(movieRuntimeDf['movie'])

    #note: frame sampling at 1fps
    #Read in images and perform kmeans for dominant colour/shade clustering

    movieFrameNumbers = dict()
    movieFrameNumbers['Hobbit 2'] = 9678
    movieFrameNumbers['Buddy'] = 5450
    movieFrameNumbers['Machete Kills'] = 6481
    movieFrameNumbers['Walter Mitty'] = 6875
    movieFrameNumbers['Paranormal Activity'] = 6065
    movieFrameNumbers['The Hunger Games-Catching Fire'] = 8775
    movieFrameNumbers['Star Wars-The Force Awakens'] = 8287
    movieFrameNumbers["Help, I Shrunk My Teacher"] = 5825
    movieFrameNumbers["I'm Off Then"] = 5331
    movieFrameNumbers["Cloudy with a Chance of Meatballs 2"] = 5690
    movieFrameNumbers['Carrie'] = 6056
    movieFrameNumbers['Walking with Dinosaurs'] = 5243
    movieFrameNumbers['Suck Me Shakespeer'] = 6747

    for movie in movieList:
        print('Processing ', movie)
        colourList = list()
        lightingList = list()
        for j in range(1, movieFrameNumbers[movie]): #movie frame numbers start at 1 
            number = '{:04d}'.format(j)
            print('Frame Number ', number)
            inputPath = "/home/sof/Notebooks/disk/Features/Movie Frames/" + movie + "/" + movie + "_" + str(number) + '.jpg'
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

            #movie frames both grey and colour loaded
            dominantColour = dominantColourOrShadeEvaluation(img)
            dominantShade = dominantColourOrShadeEvaluation(grayImg)
            colourList.append(dominantColour)
            lightingList.append(dominantShade)

        print(movie + ' save pickle object')
        dominantColourFilename = 'Pickle Objects//Colour Pickle Objects//' + movie + '.p'
        dominantShadeFilename = 'Pickle Objects//Shade Pickle Objects//' + movie + '.p' 
        pickle.dump(colourList, open(dominantColourFilename, 'wb'))
        pickle.dump(lightingList, open(dominantShadeFilename, 'wb'))



main()

