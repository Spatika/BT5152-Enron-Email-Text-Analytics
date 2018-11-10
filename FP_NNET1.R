


##### Testing nnet approach in Enron dataset

#load packages
library(caret)
library(tidyverse)
library(nnet)

#load data
data = read.csv("~/Desktop/NUS/BT5152:DMT/FP/enron_cleaned.csv", stringsAsFactors = F)
enron = data
data$poi = as.factor(data$poi)


#set seed
set.seed(1)

# Data prep
payment_data <- c('salary',
                  'bonus',
                  'long_term_incentive',
                  'deferred_income',
                  'deferral_payments',
                  'loan_advances',
                  'other',
                  'expenses',                
                  'director_fees', 
                  'total_payments')

stock_data <- c('exercised_stock_options',
                'restricted_stock',
                'restricted_stock_deferred',
                'total_stock_value')

email_data <- c("to_messages",
                'from_messages',
                'from_poi_to_this_person',
                'from_this_person_to_poi',
                'shared_receipt_with_poi')

features_list <- paste(c('poi'), payment_data , stock_data ,email_data)


#Training and testing data
enron$poi <- factor(enron$poi)
normalize <- function(x) { 
  return((x - min(x)) / (max(x) - min(x)))
}

denormalize <- function(x, min_x, max_x) { 
  return(x * (max_x - min_x) + min_x) 
}

enron_train <- enron %>% sample_frac(3/5)
enron_test <- anti_join(enron,enron_train, by = 'name') 

scaled_training <- enron_train %>% mutate_if(is.numeric, normalize)
scaled_test <- enron_test %>% mutate_if(is.numeric, normalize)


train2 = scaled_training[,-1]
test2 = scaled_test[,-1]
test2$loan_advances = 0

# Testing with nnet 
nn1 = nnet(poi ~ ., size = 2, train2)
nn2 = train(poi ~., method = 'nnet', data = train2)
summary(nn2)
plot(nn2)

nn2_pred = predict(nn2, train2)
mean(nn2_pred == scaled_training$poi)
confusionMatrix(nn2_pred, scaled_training$poi) #Training accuracy 94%

# test data pred
test_pred1 = predict(nn2, test2)
mean(test_pred1 == scaled_test$poi)
confusionMatrix(test_pred1, scaled_test$poi) #Testing accuracy 82%


# tuning the model
cvCtrl = trainControl(method="cv", number = 5)
grid = expand.grid(size = 2, decay = 0)
nn3 = train(poi ~., method = 'nnet', data = train2, trControl = cvCtrl, tuneGrid = grid, tuneLenght = 5)


nn3_pred = predict(nn3, train2)
mean(nn3_pred == scaled_training$poi)
confusionMatrix(nn3_pred, scaled_training$poi) #Training accuracy 98%

# test data pred 
test_pred2 = predict(nn3, test2)
mean(test_pred2 == scaled_test$poi)
confusionMatrix(test_pred2, scaled_test$poi) #Testing accuracy 71%


#Precision-recall testing
library(ROCR)
library(PRROC)
train_labels = scaled_training$poi
test_labels = scaled_test$poi

#Precision-recall curve for model nn2
pred = ROCR::prediction(as.numeric(test_pred1), as.numeric(test_labels))
perf1 = performance(pred, "prec", "rec")
plot(perf1) # Very low  precisions

#Precision-recall curve for model nn3
pred2 = ROCR::prediction(as.numeric(test_pred2), as.numeric(test_labels))
perf2 = performance(pred2, "prec", "rec")
plot(perf2) # Very low  precisions


