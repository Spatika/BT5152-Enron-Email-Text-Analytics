#load packages
library(caret)
library(tidyverse)

#load data
data = read.csv("enron_cleaned.csv", stringsAsFactors = F)
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
enron_test <- anti_join(enron,enron_train, by = 'X') 

scaled_training <- enron_train %>% mutate_if(is.numeric, normalize)
scaled_test <- enron_test %>% mutate_if(is.numeric, normalize)


train2 = scaled_training[,-1]
test2 = scaled_test[,-1]
test2$loan_advances = 0

# # Testing with KNN

# tuning the model
control = trainControl(method="cv", number = 5)

model_knn <- train(poi ~., data = train2,method="knn",trControl = control,
                   tuneGrid =expand.grid(k = c(3, 7, 11)))

#Training accuracy 0.89%
pred_knn_tr <- predict(model_knn,newdata =train2)
mean(pred_knn_tr == scaled_training$poi)

#Testing accuracy 0.85%
pred_knn <- predict(model_knn,newdata =test2)
mean(pred_knn == scaled_test$poi)
cm = confusionMatrix(pred_knn, scaled_test$poi)

cm$byClass["Precision"] #0.8571429
cm$byClass["Recall"] #1




