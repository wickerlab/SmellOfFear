library(ggplot2)

#get list of file names
filenames <- list.files(pattern=".csv")


#create dataframes
summaryDf <- data.frame(1,1,1,1,1)
names(summaryDf) <- c("voc", "Randomised Mean Error", "Randomised Error SD", "Unrandomised Mean Error", "Unrandomised Error SD")

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
  
  
  #T TESTS
  #two sample two tailed t-test - significant difference between the sample means 
  twoSampleTTest <- t.test(unrandomisedRMSE,randomisedRMSE)
  
  #paired two sample t test
  twoSampleTTestPaired <- t.test(unrandomisedRMSE,randomisedRMSE, paired=TRUE)
  
  #two sample non parametric Mann-Whitney U Test
  uTest <- wilcox.test(unrandomisedRMSE,randomisedRMSE)
  
  #paired non parametric test
  uTestPaired <- wilcox.test(unrandomisedRMSE,randomisedRMSE,paired=TRUE)
  
}

#drop the first placeholder row of the dataframe
summaryDf <- summaryDf[2:nrow(summaryDf),]

#convert voc to factor
summaryDf$voc = as.factor(summaryDf$voc)

#plot randomised mean and sd
pRandomised <- ggplot() +
  geom_point(data=summaryDf, aes(summaryDf$voc, summaryDf$`Randomised Mean Error`)) +
  geom_errorbar(
    data=summaryDf, 
    aes(summaryDf$voc, summaryDf$`Randomised Mean Error`, ymin=summaryDf$`Randomised Mean Error`-summaryDf$`Randomised Error SD`, ymax=summaryDf$`Randomised Mean Error`+summaryDf$`Randomised Error SD`),
    colour='red',
    width = 0.4
  )
pRandomised + xlab("VOC") + ylab("Randomised Mean Error") + ggtitle("Randomised Error")

#plot unrandomised mean and sd 
pUnrandomised <- ggplot() +
  geom_point(data=summaryDf, aes(summaryDf$voc, summaryDf$`Unrandomised Mean Error`)) +
  geom_errorbar(
    data=summaryDf, 
    aes(summaryDf$voc, summaryDf$`Unrandomised Mean Error`, ymin=summaryDf$`Unrandomised Mean Error`-summaryDf$`Unrandomised Error SD`, ymax=summaryDf$`Unrandomised Mean Error`+summaryDf$`Unrandomised Error SD`),
    colour='red',
    width = 0.4
  )
pUnrandomised + xlab("VOC") + ylab("Unrandomised Mean Error") + ggtitle("Unrandomised Error")


        

