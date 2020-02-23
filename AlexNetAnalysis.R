#########################################################
# AlexNet Type Convolutional Neural Net Analysis
# Authors: Adam Rohde & Ashley Chiu
#########################################################

library(data.table)
library(stringr)
library(car) 
source("http://www.stat.ucla.edu/~hqxu/stat201A/R/halfnormal.R") 

#################################
#################################
#################################
#Analysis of Initial Experiment

ANEx = fread('C:\\ARR\\_UCLA\\Courses\\201A Research Design, Sampling, and Analysis\\Assignments\\Project\\draft 2019-12-1\\STATS_201A_AlexNet_Data_Iter1.csv')

###########################
## EDA Boxplots
par(mfrow=c(2,2), oma = c(0, 0, 2, 0))
boxplot(ANEx$Accuracy~ANEx$Learning.Rate, main="Learning Rate", xlab = "Learning Rate", ylab =" Test Accuracy")
boxplot(ANEx$Accuracy~ANEx$Number.of.Epochs, main="Number of Epochs", xlab = "Epochs", ylab =" Test Accuracy")
boxplot(ANEx$Accuracy~ANEx$Batch.Size, main="Batch Size", xlab = "Batch Size", ylab =" Test Accuracy")
boxplot(ANEx$Accuracy~ANEx$Dropout, main="No Dropout or Dropout", xlab = "Dropout", ylab =" Test Accuracy")
mtext("Exploratory Boxplots - Initial Experiment", outer = TRUE, cex = 1.5, )

par(mfrow=c(2,2), oma = c(0, 0, 2, 0))
boxplot(ANEx$Accuracy~ANEx$Activation.Function, main="Activation Function", xlab = "Activation Function", ylab =" Test Accuracy")
boxplot(ANEx$Accuracy~ANEx$Additional.Convolution.Layer, main="Additional Conv. Layer", xlab = "Additional Conv. Layer", ylab =" Test Accuracy")
boxplot(ANEx$Accuracy~ANEx$Normalization, main="No Normalization or Normalization", xlab = "Normalization", ylab =" Test Accuracy")
boxplot(ANEx$Accuracy~ANEx$Block, main="Block", xlab= "Block", ylab="Test Accuracy")

###########################
#All Data from Initial Experiment

#Main Effect
MEmodel = lm(Accuracy ~ (A+B+C+D+E+F+G)+Block,data=ANEx)
estimates = MEmodel$coefficients*2
par(mfrow=c(1,1))
halfnormalplot(estimates[-1],l=T,n=7, main = "Half-Normal Plot of ME")
summary(MEmodel)
par(mfrow=c(1,2))
plot(MEmodel,1:2)

#Steps
Stepmodel = lm(Accuracy ~ (A+B+C+D+E+F+G)^2+Block,data=ANEx)
SelectedModel <- step(Stepmodel,scope = list(upper=~.,lower=~1))
summary(SelectedModel)
estimates = SelectedModel$coefficients*2
par(mfrow=c(1,1))
halfnormalplot(estimates[-1],l=T,n=7, main ="Half-Normal Plot of ME and 2FI")

#final model
Finalmodel = lm(Accuracy ~ A+B+C+D+E+F+G+B:D+C:E+C:F+C:G+Block,data=ANEx)
summary(Finalmodel)
par(mfrow=c(1,2))
plot(Finalmodel,1:2)

###########################
#Remove Outliers from Initial Experiment
SummaryAcc = summary(ANEx$Accuracy)
IQR = IQR(ANEx$Accuracy)

ANExREMOVE<-ANEx[ANEx$Accuracy<=SummaryAcc[2]-1.5*IQR | ANEx$Accuracy>=SummaryAcc[5]+1.5*IQR ,]
ANExLIMITED<-ANEx[ANEx$Accuracy>SummaryAcc[2]-1.5*IQR & ANEx$Accuracy<SummaryAcc[5]+1.5*IQR,]

#Main Effect
par(mfrow=c(1,1))
MEmodel2 = lm(Accuracy ~ (A+B+C+D+E+F+G)+Block,data=ANExLIMITED)
estimates = MEmodel2$coefficients*2
halfnormalplot(estimates[-1],l=T,n=7, main = "Half-Normal Plot - ME, Outliers Removed")
summary(MEmodel2)

#Steps
Stepmodel2 = lm(Accuracy ~ (A+B+C+D+E+F+G)^2+Block,data=ANExLIMITED)
SelectedModel2 <- step(Stepmodel2,scope = list(upper=~.,lower=~1))
summary(SelectedModel2)
estimates2 = SelectedModel2$coefficients*2
halfnormalplot(estimates2[-1],l=T,n=7, main = "Half-Normal Plot- ME and 2FI, Outliers Removed")
par(mfrow=c(1,2))
plot(SelectedModel2,1:2)

#Log transform
Stepmodel.log = lm(log(Accuracy) ~ (A+B+C+D+E+F+G)^2+Block,data=ANExLIMITED)
SelectedModel.log <- step(Stepmodel.log,scope = list(upper=~.,lower=~1))
summary(SelectedModel.log)
estimates.log = SelectedModel.log$coefficients*2
par(mfrow=c(1,1))
halfnormalplot(estimates.log[-1],l=T,n=7, main = "Half-Normal Plot - ME and 2FI, Outliers Removed + Log Transform")
par(mfrow=c(1,2))
plot(SelectedModel.log,1:2)

#################################
#################################
#################################
#Analysis of Follow Up Experiment

ANEx = fread('C:\\ARR\\_UCLA\\Courses\\201A Research Design, Sampling, and Analysis\\Assignments\\Project\\draft 2019-12-1\\STATS_201A_AlexNet_Data_Iter2.csv')

###########################
#All Data from Follow Up Experiment

#Main Effect
par(mfrow=c(1,1))
MEmodel = lm(log(Accuracy) ~ (A+B+C+D+E+F+G)+Block,data=ANEx)
estimates = MEmodel$coefficients*2
halfnormalplot(estimates[-1],l=T,n=7, main = "Half-Normal Plot of ME (Follow-up Experiment)")
summary(MEmodel)

#Steps
Stepmodel = lm(log(Accuracy) ~ (A+B+C+D+E+F+G)^2+Block,data=ANEx)
SelectedModel <- step(Stepmodel,scope = list(upper=~.,lower=~1))
summary(SelectedModel)
estimates = SelectedModel$coefficients*2
halfnormalplot(estimates[-1],l=T,n=7, main= "Half-Normal Plot - ME and 2FI with Log Transform (Follow-up Experiment)")
par(mfrow=c(1,2))
plot(SelectedModel,1:2)

#Best Factor Settings
ANEx$predicted = exp(SelectedModel$fitted.values)
maxPredicted = ANEx[order(with(ANEx,-predicted)),]
maxPredicted = maxPredicted[1,]
print(maxPredicted)