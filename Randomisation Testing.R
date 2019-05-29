#get list of file names
filenames <- list.files(pattern=".csv")

#create results dataframe
resultsDf <- data.frame()

pThreshold <- 0.05

cat("VOC", "\t", "T Test","\t", "Paired T Test","\t", "U Test", "\t","Paired U Test","\n")

for (filename in filenames){
  #read csv and extract required data
  vocData <- read.csv(filename, header=TRUE,sep=",",nrows=200, row.names=1)
  unrandomisedRMSE <- vocData[seq(from=1, to=199, by=2),3]
  randomisedRMSE <- vocData[seq(from=2, to=200, by=2),3]
  voc <- as.character(vocData[1,'VOC'])
  

  #two sample two tailed t-test - significant difference between the sample means 
  twoSampleTTest <- t.test(unrandomisedRMSE,randomisedRMSE)
  
  #paired two sample t test
  twoSampleTTestPaired <- t.test(unrandomisedRMSE,randomisedRMSE, paired=TRUE)
  
  #two sample non parametric Mann-Whitney U Test
  uTest <- wilcox.test(unrandomisedRMSE,randomisedRMSE)
  
  #paired non parametric test
  uTestPaired <- wilcox.test(unrandomisedRMSE,randomisedRMSE,paired=TRUE)
  
  cat(voc,"\t", twoSampleTTest$p.value,"\t", twoSampleTTestPaired$p.value,"\t", uTest$p.value,"\t",uTestPaired$p.value, "\n")

}
