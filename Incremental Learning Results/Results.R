library(ggplot2)
library(gtools)

#read rmse files
filenames <- list.files(pattern=".csv")
rmseDataRHT <- read.csv(filenames[1], header=TRUE, sep=",")
colnames(rmseDataRHT) = c("Instance", "RMSE")

#plot 
p <- ggplot() +
  geom_point(data=rmseDataRHT, aes(rmseDataRHT$Instance, rmseDataRHT$RMSE), colour='green')

p + xlab("Instance Number") + ylab("RMSE") + ggtitle("Hoeffding Tree RMSE")