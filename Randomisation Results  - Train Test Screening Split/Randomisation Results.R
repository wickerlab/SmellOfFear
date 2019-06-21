library(ggplot2)
library(gtools)

#get list of file names and sort the list of filenames into the desired order
filenames <- list.files(pattern=".csv")

#create dataframes
summaryDf <- data.frame(1,1,1,1,1)
names(summaryDf) <- c("voc", "Randomised Mean Error", "Randomised Error SD", "Unrandomised Mean Error", "Unrandomised Error SD")
resultsDf <- data.frame(1,1,1,1,1,1,1)
names(resultsDf) <- c("voc", 
                      "Two Sample T Test", "Two Sample Non-parametric Test",
                      "Paired T Test", "Paired Non-parametric Test", 
                      "One Sample T Test", "One Sample Non-parametric Test")

#0.05 cut off and box plot or bar plot


pThreshold <- 0.05

for (filename in filenames){
  #read csv and extract required data
  vocData <- read.csv(filename, header=TRUE,sep=",",nrows=200, row.names=1)
  unrandomisedRMSE <- vocData[seq(from=1, to=199, by=2),3]
  randomisedRMSE <- vocData[seq(from=2, to=200, by=2),3]
  voc <- as.character(vocData[1,'VOC'])
 
  
  #CALCULATE SUMMARY STATS
  randomisedErrorMean <- mean(randomisedRMSE)
  unrandomisedErrorMean <- mean(unrandomisedRMSE)
  randomisedSd <- sd(randomisedRMSE)
  unrandomisedSd <- sd(unrandomisedRMSE)
  tempDf <- data.frame(voc, randomisedErrorMean,randomisedSd,unrandomisedErrorMean,unrandomisedSd)
  names(tempDf) <- c("voc", "Randomised Mean Error", "Randomised Error SD", "Unrandomised Mean Error", "Unrandomised Error SD")
  summaryDf <- rbind(summaryDf, tempDf)
  
  #TWO SAMPLE TEST
  #two sample two tailed t-test - significant difference between the sample means 
  twoSampleTTest <- t.test(unrandomisedRMSE,randomisedRMSE)
  #two sample non parametric Mann-Whitney U Test
  uTest <- wilcox.test(unrandomisedRMSE,randomisedRMSE)
  
  #PAIRED TEST
  #paired two sample t test
  twoSampleTTestPaired <- t.test(unrandomisedRMSE,randomisedRMSE, paired=TRUE)
  #paired non parametric test
  uTestPaired <- wilcox.test(unrandomisedRMSE,randomisedRMSE,paired=TRUE)
  
  #ONE SAMPLE ONE TAIL TEST
  meanUnrandomisedRMSE <- mean(unrandomisedRMSE)
  #t test
  oneSampleTTest <- t.test(randomisedRMSE, mu=meanUnrandomisedRMSE, alternative="greater")
  #non parametric
  oneSampleUTest <- wilcox.test(randomisedRMSE,mu=meanUnrandomisedRMSE, alternative="greater")
  tempDf <- data.frame(voc, 
                       twoSampleTTest$p.value, uTest$p.value,
                       twoSampleTTestPaired$p.value, uTestPaired$p.value,
                       oneSampleTTest$p.value, oneSampleUTest$p.value)
  names(tempDf) <- c("voc", 
                        "Two Sample T Test", "Two Sample Non-parametric Test",
                        "Paired T Test", "Paired Non-parametric Test", 
                        "One Sample T Test", "One Sample Non-parametric Test")
  resultsDf <- rbind(resultsDf, tempDf)
  
}

#drop the first placeholder row of the dataframe
summaryDf <- summaryDf[2:nrow(summaryDf),]
resultsDf <- resultsDf[2:nrow(resultsDf),]

#convert voc to factor
summaryDf$voc = as.factor(summaryDf$voc)

#plot mean and sd
p <- ggplot() +
  geom_point(data=summaryDf, aes(summaryDf$voc, summaryDf$`Randomised Mean Error`), colour='green') +
  geom_errorbar(
    data=summaryDf, 
    aes(summaryDf$voc, summaryDf$`Randomised Mean Error`, ymin=summaryDf$`Randomised Mean Error`-summaryDf$`Randomised Error SD`, ymax=summaryDf$`Randomised Mean Error`+summaryDf$`Randomised Error SD`),
    colour='red',
    width = 0.4
  ) + 
  geom_point(data=summaryDf, aes(summaryDf$voc, summaryDf$`Unrandomised Mean Error`), colour='blue') +
  geom_errorbar(
    data=summaryDf, 
    aes(summaryDf$voc, summaryDf$`Unrandomised Mean Error`, ymin=summaryDf$`Unrandomised Mean Error`-summaryDf$`Unrandomised Error SD`, ymax=summaryDf$`Unrandomised Mean Error`+summaryDf$`Unrandomised Error SD`),
    colour='red',
    width = 0.4
  )
  
p + xlab("VOC") + ylab("RMSE") + ggtitle("Randomised and Unrandomised Error")


        

