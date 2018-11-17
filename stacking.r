
library(caretEnsemble)
library(caret)

TL=list(
   m1=caretModelSpec(method='knn'),
   m2=caretModelSpec(method='rpart'),
   m3=caretModelSpec(method='ranger'))

set.seed(5152)
folds = createFolds(train_poi$poi, k = 5)

set.seed(5152)
stack_control = trainControl(method='repeatedcv', number=5, repeats=3, index=folds, savePredictions='final', classProbs=TRUE, summaryFunction=twoClassSummary, sampling = "up")

models = caretList(poi ~ ., data = cbind(train_features_df,train_poi), metric = 'ROC', trControl=stack_control, tuneList = TL)

results = resamples(models)
summary(results)

stack.glm = caretStack(models, method="glm", metric="ROC", trControl=stack_control)
print(stack.glm)
  
stack_pred = predict(stack.glm, train_features_df)
confusionMatrix(stack_pred, train_poi$poi, mode = 'everything', positive = "True")

stack_pred_test = predict(stack.glm, test_features_df)
confusionMatrix(stack_pred_test,test_poi$poi, mode = 'everything', positive = "True")

### AUC
test_pred_stack <- ROCR::prediction(as.numeric(predict(stack.glm, test_features_df)), as.numeric(test_poi$poi))
as.numeric(performance(test_pred_stack, "auc")@y.values)