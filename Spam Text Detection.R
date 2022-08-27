######
# AUTHOR: Harshitha Nagapudi, Oluwabusayo Oyekanmi, Elham Inamdar, Hassan Wasswa
# FILENAME: CS5331 - ML - Assignment3 - Source Code.R
# SPECIFICATION: To implement the proposal submitted for Assignment 2 on an 
#                 information security problem to solve with machinelearning.
# FOR: CS 5331 Machine Learning and Information Security section 001/003
#######



#import required libraries
library(neuralnet)
library(caret)
library(dplyr)
library(ggplot2)
library("psych")
library("e1071")
library(gmodels)

#set working directory
setwd("C:/spambase")

#read data
df <- read.csv("spambase.csv")

#set column names
names(df)<-make.names(c("word_freq_make","word_freq_address","word_freq_all","word_freq_3d","word_freq_our","word_freq_over",
                        "word_freq_remove","word_freq_internet","word_freq_order","word_freq_mail","word_freq_receive",
                        "word_freq_will","word_freq_people","word_freq_report","word_freq_addresses","word_freq_free",
                        "word_freq_business","word_freq_email","word_freq_you","word_freq_credit","word_freq_your","word_freq_font"
                        ,"word_freq_000","word_freq_money","word_freq_hp","word_freq_hpl","word_freq_george","word_freq_650","word_freq_lab",
                        "word_freq_labs","word_freq_telnet","word_freq_857","word_freq_data","word_freq_415","word_freq_85","word_freq_technology",
                        "word_freq_1999","word_freq_parts","word_freq_pm","word_freq_direct","word_freq_cs","word_freq_meeting","word_freq_original",
                        "word_freq_project","word_freq_re","word_freq_edu","word_freq_table","word_freq_conference","char_freq_semicolon",
                        "char_freq_leftparenthes","char_freq_leftbracket","char_freq_approximation","char_freq_dolar","char_freq_hash",
                        "capital_run_length_average","capital_run_length_longest","capital_run_length_total","label"))

##inspect data structure, size and distribution
#data set shape

dim(df)
#convert label from numeric to categorical
df["cat_label"] <-df$label
#data structure and summary
df$cat_label <- as.factor(df$cat_label)
str(df)
summary(df)
#distribution of classes in terms of number and proportion
table(df$cat_label)
prop.table(table(df$cat_label))


#visualization of instance distribution


#Divide the data set into  80% train and 20% test set
set.seed(1273)
index <- createDataPartition(df$cat_label, p = 0.8, list=F)

train_df <- df[index,]
test_df <- df[-index,]

#Perform PCA using the prcomp function
set.seed(145)
train_df.pca <- prcomp(train_df[,1:57],
                       center = T, scale. = T)
#inspect the eigen vectors
train_df.pca

# get eigen values
eigen.values <- (train_df.pca$sdev)^2
eigen.values

# transform train set using eigen vectors to generate principle components
train_set <- predict(train_df.pca, train_df[,1:57])

test_set <- predict(train_df.pca, test_df[,1:57])

#Select first 30 components. These explain 73.55% of variance
train_set_30 <- train_set[,1:30]

#form training set with top 30 principal components by concatenating with label
train_set_30 <- cbind(label=train_df[,59], as.data.frame(train_set_30))

train_set_30$label = as.factor(train_set_30$label)


####Naive Bayes####
#train naive bayes model using k-fold cross-validation
# with k = 10
train_control <- trainControl(method="cv", number=10, savePredictions=T)
#naive bayes model based on cross validation with 10 folds
set.seed(222)
nb_model_cv <- train(label~., data = train_set_30, method = 'naive_bayes',
                     trControl=train_control)


#transform test set for for model testing using top 30 principal components
test_set <- predict(train_df.pca, test_df[,1:57])
test_set_30 <- test_set[,1:30]
test_set_30 <- cbind(label=test_df[,59], as.data.frame(test_set_30))
test_set_30$label = as.factor(test_set_30$label)

# generate predictions on the test set
pred_30_pca <- predict(nb_model_cv, newdata = test_set_30)

# get confusion matrix
conf_mat_30_pca = confusionMatrix(pred_30_pca, test_set_30$label,positive='1')
conf_mat_30_pca

accuracy_30_pca = 100*sum(diag(conf_mat_30_pca$table))/sum(conf_mat_30_pca$table)

#Training on 35 components. These explain 80.6% of variance

train_set_35 <- train_set[,1:35]

train_set_35 <- cbind(label=train_df[,59], as.data.frame(train_set_35))
train_set_35$label = as.factor(train_set_35$label)


#Train naive bayes model based on cross validation with 10 folds
set.seed(333)
nb_model_cv <- train(label~., data = train_set_35, method = 'naive_bayes',
                     trControl=train_control)

# generate predictions on the test set
test_set_35 <- test_set[,1:35]
test_set_35 <- cbind(label=test_df[,59], as.data.frame(test_set_35))
test_set_35$label = as.factor(test_set_35$label)
pred_35_pca <- predict(nb_model_cv, newdata = test_set_35)

# get confusion matrix
conf_mat_35_pca = confusionMatrix(pred_35_pca, test_set_35$label,positive='1')
conf_mat_35_pca

accuracy_35_pca = 100*sum(diag(conf_mat_35_pca$table))/sum(conf_mat_35_pca$table)

#Train on 40 components. These explain 86.87% of variance

train_set_40 <- train_set[,1:40]
train_set_40 <- cbind(label=train_df[,59], as.data.frame(train_set_40))
train_set_40$label = as.factor(train_set_40$label)

#naive bayes model based on cross validation with 10 folds
set.seed(444)
nb_model_cv <- train(label~., data = train_set_40, method = 'naive_bayes',
                     trControl=train_control)

# generate predictions on the test set with 40 components
test_set_40 <- test_set[,1:40]
test_set_40 <- cbind(label=test_df[,59], as.data.frame(test_set_40))
test_set_40$label = as.factor(test_set_40$label)
pred_40_pca <- predict(nb_model_cv, newdata = test_set_40)

#get confusion matrix
conf_mat_40_pca = confusionMatrix(pred_40_pca, test_set_40$label,positive='1')
conf_mat_40_pca

accuracy_40_pca = 100*sum(diag(conf_mat_40_pca$table))/sum(conf_mat_40_pca$table)

#Training on 45 components. These explain 92.23% of variance

train_set_45 <- train_set[,1:45]
train_set_45 <- cbind(label=train_df[,59], as.data.frame(train_set_45))
train_set_45$label = as.factor(train_set_45$label)

#naive bayes model based on cross validation with 10 folds
set.seed(555)
nb_model_cv <- train(label~., data = train_set_45, method = 'naive_bayes',
                     trControl=train_control)

# generate predictions on the test set using 45 components
test_set_45 <- test_set[,1:45]
test_set_45 <- cbind(label=test_df[,59], as.data.frame(test_set_45))
test_set_45$label = as.factor(test_set_45$label)
pred_45_pca <- predict(nb_model_cv, newdata = test_set_45)

# get confusion matrix
conf_mat_45_pca = confusionMatrix(pred_45_pca, test_set_45$label,positive='1')
conf_mat_45_pca

accuracy_45_pca = 100*sum(diag(conf_mat_45_pca$table))/sum(conf_mat_45_pca$table)


num_of_comp <-as.factor(c(30,35,40,45))
accu <- c(accuracy_30_pca, accuracy_35_pca, accuracy_40_pca, accuracy_45_pca)


plot(x=num_of_comp, y=accu, xlab="number of components", ylab="Accuracy")


#####Neural network#####
#with 30 components

#Select first 30 components. These explain 73.55% of variance
train_set_30 <- train_set[,1:30]

#form training set with top 30 principal components by concatenating with label
train_set_30 <- cbind(label=train_df[,58], as.data.frame(train_set_30))
set.seed(253)
nn_30 <- neuralnet(label~., data = train_set_30, hidden=5, rep=10, threshold = 0.1,
                stepmax = 1e6, learningrate=1e-4, err.fct = "ce", linear.output = F)
plot(nn_30)

test_set_30 <- test_set[,1:30]

#form training set with top 30 principal components by concatenating with label
test_set_30 <- cbind(label=test_df[,58], as.data.frame(test_set_30))


Predict_30=neuralnet::compute(nn_30,test_set_30[,-1])


prob_30 <- Predict_30$net.result
pred_30 <- ifelse(prob_30>0.5, 1, 0)


conf_mat_30_nn <- confusionMatrix(as.factor(pred_30), as.factor(test_set_30$label), positive = "1")

accu_30_nn <- 100*sum(diag(conf_mat_30_nn$table))/sum(conf_mat_30_nn$table) 
accu_30_nn


#with 35 components

#Select first 35 components. These explain 80.6% of variance
train_set_35 <- train_set[,1:35]

#form training set with top 35 principal components by concatenating with label
train_set_35 <- cbind(label=train_df[,58], as.data.frame(train_set_35))
set.seed(253)
nn_35 <- neuralnet(label~., data = train_set_35, hidden=5, rep=10, threshold = 0.1,
                   stepmax = 1e6, learningrate=1e-4, err.fct = "ce", linear.output = F)
plot(nn_35)

test_set_35 <- test_set[,1:35]

#form training set with top 35 principal components by concatenating with label
test_set_35 <- cbind(label=test_df[,58], as.data.frame(test_set_35))


Predict_35=neuralnet::compute(nn_35,test_set_35[,-1])


prob_35 <- Predict_35$net.result
pred_35 <- ifelse(prob_35>0.5, 1, 0)


conf_mat_35_nn <- confusionMatrix(as.factor(pred_35), as.factor(test_set_35$label), positive = "1")

accu_35_nn <- 100*sum(diag(conf_mat_35_nn$table))/sum(conf_mat_35_nn$table) 
accu_35_nn


#with 40 components

#Select first 40 components. These explain 86.87% of variance
train_set_40 <- train_set[,1:40]

#form training set with top 40 principal components by concatenating with label
train_set_40 <- cbind(label=train_df[,58], as.data.frame(train_set_40))
set.seed(253)
nn_40 <- neuralnet(label~., data = train_set_40, hidden=5, rep=10, threshold = 0.1,
                   stepmax = 1e6, learningrate=1e-4, err.fct = "ce", linear.output = F)
plot(nn_40)

test_set_40 <- test_set[,1:40]

#form training set with top 40 principal components by concatenating with label
test_set_40 <- cbind(label=test_df[,58], as.data.frame(test_set_40))


Predict_40=neuralnet::compute(nn_40,test_set_40[,-1])


prob_40 <- Predict_40$net.result
pred_40 <- ifelse(prob_40>0.5, 1, 0)


conf_mat_40_nn <- confusionMatrix(as.factor(pred_40), as.factor(test_set_40$label), positive = "1")

accu_40_nn <- 100*sum(diag(conf_mat_40_nn$table))/sum(conf_mat_40_nn$table) 
accu_40_nn



#with 45 components

#Select first 45 components. These explain 92.23% of variance
train_set_45 <- train_set[,1:45]

#form training set with top 45 principal components by concatenating with label
train_set_45 <- cbind(label=train_df[,58], as.data.frame(train_set_45))
set.seed(253)
nn_45 <- neuralnet(label~., data = train_set_45, hidden=5, rep=10, threshold = 0.1,
                   stepmax = 1e6, learningrate=1e-4, err.fct = "ce", linear.output = F)
plot(nn_45)

test_set_45 <- test_set[,1:45]

#form training set with top 45 principal components by concatenating with label
test_set_45 <- cbind(label=test_df[,58], as.data.frame(test_set_45))


Predict_45=neuralnet::compute(nn_45,test_set_45[,-1])


prob_45 <- Predict_45$net.result
pred_45 <- ifelse(prob_45>0.5, 1, 0)


conf_mat_45_nn <- confusionMatrix(as.factor(pred_45), as.factor(test_set_45$label), positive = "1")

accu_45_nn <- 100*sum(diag(conf_mat_45_nn$table))/sum(conf_mat_45_nn$table) 
accu_45_nn

###### Using Naive Bayes model for prediction with full dataset

#Read the data
data <- read.csv("spambase.csv", header=TRUE)
set.seed(451)

#Splitting Train and Test Data
dt = sort(sample(nrow(data), nrow(data)*.8))
train<-data[dt,]
test<-data[-dt,]

#Checking the distribution of 0 and 1 in Train and Test data
prop.table(table(train$Column58))
prop.table(table(test$Column58))

#Naive Bayes
spam_Model = naiveBayes(train, train$Column58)
test.predicted = predict(spam_Model, test)

CrossTable(test.predicted,
           test$Column58,
           prop.chisq = FALSE,
           prop.t     = FALSE,
           dnn        = c("predicted", "actual"))
spam_test_pred <- predict(spam_Model, test)
table(spam_test_pred)
table(test$Column58)
confusionMatrix(spam_test_pred, as.factor(test$Column58), dnn = c("predicted", "actual"))


#Improved Naive Bayes
improved_spam_Model = naiveBayes(train, train$Column58, laplace = 1)
test.predicted_improved = predict(improved_spam_Model, test)
CrossTable(test.predicted_improved,
           test$Column58,
           prop.chisq = FALSE,
           prop.t     = FALSE,
           dnn        = c("predicted", "actual"))

spam_test_pred_impr <- predict(improved_spam_Model, test)
table(spam_test_pred)
table(test$Column58)
confusionMatrix(spam_test_pred_impr, as.factor(test$Column58), dnn = c("predicted", "actual"))

#nfolds = 5
nfolds = 10
#Randomly shuffle the data
newdata<-data[sample(nrow(data)),]
#Create folds (makes a vector marking each fold, such as 1 1 1 2 2 2 3 3 3, for 3 folds)
folds <- cut(seq(1,nrow(newdata)),breaks=nfolds,labels=FALSE)
average_accuracy = 0.0
for(i in 1:nfolds){
  #Choose 1 fold and separate the Train and Test Data
  testIndexes <- which(folds==i,arr.ind=TRUE)
  testData <- data[testIndexes, ]
  trainData <- data[-testIndexes, ]
  #Build the model using Naive Bayes
  spam_Model = naiveBayes(trainData, trainData$Column58)
  testData$Predict = predict(spam_Model, testData)
  
  x = table(testData$Predict, testData$Column58)
  if (ncol(x) == 1) {
    if (colnames(x)[1] == "0") {
      x = cbind(x,"1"=c(0,0))
    } else {
      x = cbind("0"=c(0,0),x)
    }
  }
  x = x[2:1,2:1]
  accuracy = (x[1,1]+x[2,2])/nrow(testData)
  average_accuracy = average_accuracy + accuracy
  print(paste("Iteration i: ",i))
  print(x)
  print(paste("Accuracy: ",accuracy))
  print("--------------------")
}
print(paste("Average Accuracy: ",average_accuracy/nfolds))

accu_full_nb = average_accuracy/nfolds


##### Neural Network with full Dataset

#Read the data
data <- read.csv("spambase.csv", header=TRUE)

#Splitting Train and Test Data
dt = sort(sample(nrow(data), nrow(data)*.8))
train<-data[dt,]
test<-data[-dt,]

#Checking the distribution of 0 and 1 in Train and Test data
prop.table(table(train$Column58))
prop.table(table(test$Column58))

nn <- neuralnet(Column58~., data = train, hidden=5, rep=10, threshold = 0.1,
                stepmax = 1e6, learningrate=1e-4, err.fct = "ce", linear.output = F)
plot(nn)

Predict=neuralnet::compute(nn,test)
Predict$net.result

prob <- Predict$net.result
pred <- ifelse(prob>0.5, 1, 0)

conf_mat <- confusionMatrix(as.factor(pred), as.factor(test$Column58), positive = "1")

accu_full_nn <- 100*sum(diag(conf_mat$table))/sum(conf_mat$table)
accu_full_nn


########Performance Plots

cols <- c("number_of_features", "Naive_Bayes", "Neural_Network")
all_features <- c("All_features", accu_full_nb, accu_full_nn)
pca_30 <- c("PCA_30_components", accuracy_30_pca, accu_30_nn)
pca_35 <- c("PCA_35_components", accuracy_35_pca, accu_35_nn)
pca_40 <- c("PCA_40_components", accuracy_40_pca, accu_40_nn)
pca_45 <- c("PCA_45_components", accuracy_45_pca, accu_45_nn)


results <- as.data.frame(matrix(data= c(all_features, pca_30, pca_35, pca_40, pca_45), nrow = 5, byrow = T))

names(results)<-make.names(cols)

#individual performance plots
library(ggplot2)

ggplot(results, 
       aes(x=number_of_features,y=Naive_Bayes))+
  geom_bar(stat="identity")

ggplot(results, 
       aes(x=number_of_features,y=NeuralNetwork))+
         geom_bar(stat="identity")

#comparison plots
dat_results <- data.frame(
  Naive_Bayes = c(accu_full_nb, accuracy_30_pca, accuracy_35_pca, accuracy_40_pca,
                  accuracy_45_pca),
  Neural_Network = c(accu_full_nn, accu_30_nn, accu_35_nn, accu_40_nn, accu_45_nn),
  number_of_features = as.factor(c("All_features", "PCA_30_components","PCA_35_components",
                                   "PCA_40_components", "PCA_35_components"))
)

dat_long <- dat_results %>%
  gather("Stat", "Value", -number_of_features)

ggplot(dat_long, aes(x = number_of_features, y = Value,ylim(70,100), fill = Stat)) +
  geom_col(position = "dodge")
