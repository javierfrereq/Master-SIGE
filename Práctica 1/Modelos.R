library(caret)
library(tidyverse)
library(funModeling)
library(pROC)
library(partykit)
library(randomForest)
library(DMwR)

## ---------------------------------------------------------------
##Funcion para pintar la curva Roc
my_roc <- function(data, predictionProb, target_var, positive_class) {
  auc <- roc(data[[target_var]], predictionProb[[positive_class]], levels = unique(data[[target_var]]))
  roc <- plot.roc(auc, ylim=c(0,1), type = "S" , print.thres = T, main=paste('AUC:', round(auc$auc[[1]], 2)))
  return(list("auc" = auc, "roc" = roc))
}

## -------------------------------------------------------------------------------------
##Leemos el .CSV
# Usando "P_LoanStats_2017Q4.csv"
# Variable de clasificacion: loan_status
data_raw <- read_csv('P_LoanStats_2017Q4.csv')
data <- data_raw %>%
  na.exclude() %>%
  mutate(loan_status = as.factor(loan_status))

ggplot(data) + geom_histogram(aes(x = loan_status, fill = loan_status), stat = 'count')

set.seed(0)

## 1. Crear modelo de prediccion usando rpart
trainIndex <- createDataPartition(data$loan_status, p = .82, list = FALSE, times = 1)
train <- data[ trainIndex, ] 
val   <- data[-trainIndex, ]

# Entrenar modelo
rpartCtrl <- trainControl(verboseIter = F, classProbs = TRUE, summaryFunction = twoClassSummary)
rpartParametersGrid <- expand.grid(.cp = c(0.001, 0.01, 0.1, 0.5))
rpartModel <- train(loan_status ~ ., data = train, method = "rpart", metric = "ROC", 
                    trControl = rpartCtrl, tuneGrid = rpartParametersGrid)

# Validacion
prediction     <- predict(rpartModel, val, type = "raw")
predictionProb <- predict(rpartModel, val, type = "prob")

auc <- roc(val$loan_status, predictionProb[["Unpaid"]], levels = unique(val[["loan_status"]]))
roc_validation <- plot.roc(auc, ylim=c(0,1), type = "S" , print.thres = T, 
                           main=paste('[Modelo Rpart] Validation AUC:', round(auc$auc[[1]], 2)))


# Obtener valores de accuracy, precision, recall, f-score (manualmente)
results <- cbind(val, prediction)
results <- results %>%
  mutate(contingency = as.factor(
    case_when(
      loan_status == 'Unpaid' & prediction == 'Unpaid' ~ 'TP',
      loan_status == 'Paid'  & prediction == 'Unpaid' ~ 'FP',
      loan_status == 'Paid'  & prediction == 'Paid'  ~ 'TN',
      loan_status == 'Unpaid' & prediction == 'Paid'  ~ 'FN'))) 
TP <- length(which(results$contingency == 'TP'))
TN <- length(which(results$contingency == 'TN'))
FP <- length(which(results$contingency == 'FP'))
FN <- length(which(results$contingency == 'FN'))
n  <- length(results$contingency)

table(results$contingency) # comprobar recuento de TP, TN, FP, FN

accuracy <- (TP + TN) / n
error <- (FP + FN) / n

precision   <- TP / (TP + FP)
sensitivity <- TP / (TP + FN)
specificity <- TN / (TN + FP)
f_measure   <- (2 * TP) / (2 * TP + FP + FN)
f_measure

## -------------------------------------------------------------------------------------

# Otro modelo utilizando rpart con cross-validation
rpartCtrl_2 <- trainControl(
  verboseIter = F, 
  classProbs = TRUE, 
  method = "repeatedcv",
  number = 10,
  repeats = 1,
  summaryFunction = twoClassSummary)
rpartModel_2 <- train(loan_status ~ ., data = train, method = "rpart", metric = "ROC", 
                      trControl = rpartCtrl_2, tuneGrid = rpartParametersGrid)

# Validacion
prediction     <- predict(rpartModel_2, val, type = "raw")
predictionProb <- predict(rpartModel_2, val, type = "prob")

auc <- roc(val$loan_status, predictionProb[["Unpaid"]], levels = unique(val[["loan_status"]]))
roc_validation <- plot.roc(auc, ylim=c(0,1), type = "S" , print.thres = T, 
                           main=paste('[Modelo Rpart cross-validation] Validation AUC:', round(auc$auc[[1]], 2)))


print(rpartModel_2)
varImp(rpartModel_2)
dotPlot(varImp(rpartModel_2))

plot(rpartModel_2)
plot(rpartModel_2$finalModel)
text(rpartModel_2$finalModel)

partyModel_2 <- as.party(rpartModel_2$finalModel)
plot(partyModel_2, type = 'simple')

# Obtener valores de accuracy, precision, recall, f-score (manualmente)
results <- cbind(val, prediction)
results <- results %>%
  mutate(contingency = as.factor(
    case_when(
      loan_status == 'Unpaid' & prediction == 'Unpaid' ~ 'TP',
      loan_status == 'Paid'  & prediction == 'Unpaid' ~ 'FP',
      loan_status == 'Paid'  & prediction == 'Paid'  ~ 'TN',
      loan_status == 'Unpaid' & prediction == 'Paid'  ~ 'FN'))) 
TP <- length(which(results$contingency == 'TP'))
TN <- length(which(results$contingency == 'TN'))
FP <- length(which(results$contingency == 'FP'))
FN <- length(which(results$contingency == 'FN'))
n  <- length(results$contingency)

table(results$contingency) # comprobar recuento de TP, TN, FP, FN

accuracy <- (TP + TN) / n
error <- (FP + FN) / n

precision   <- TP / (TP + FP)
sensitivity <- TP / (TP + FN)
specificity <- TN / (TN + FP)
f_measure   <- (2 * TP) / (2 * TP + FP + FN)
f_measure


## -------------------------------------------------------------------------------------
## 2. Prediccion Arboles-Aleatorios

## Crear modelo de prediccionn usando RF
### Modelo basico, ajuste de manual de hiperparametros (.mtry)
rfCtrl <- trainControl(verboseIter = F, classProbs = TRUE, method = "repeatedcv", 
                       number = 10, repeats = 1, summaryFunction = twoClassSummary)
rfParametersGrid <- expand.grid(.mtry = c(sqrt(ncol(train))))
rfModel <- train(loan_status ~ ., data = train, method = "rf", metric = "ROC", 
                 trControl = rfCtrl,tuneGrid = rfParametersGrid)
my_roc(val, predict(rfModel, val, type = "prob"), "loan_status", "Unpaid")

# Validacion
prediction     <- predict(rfModel, val, type = "raw")
predictionProb <- predict(rfModel, val, type = "prob")

auc <- roc(val$loan_status, predictionProb[["Unpaid"]], levels = unique(val[["loan_status"]]))
roc_validation <- plot.roc(auc, ylim=c(0,1), type = "S" , print.thres = T, main=paste('[Modelo RF] Validation AUC:', round(auc$auc[[1]], 2)))

#Independiente
# Obtener valores de accuracy, precision, recall, f-score (manualmente)
results <- cbind(val, prediction)
results <- results %>%
  mutate(contingency = as.factor(
    case_when(
      loan_status == 'Unpaid' & prediction == 'Unpaid' ~ 'TP',
      loan_status == 'Paid'  & prediction == 'Unpaid' ~ 'FP',
      loan_status == 'Paid'  & prediction == 'Paid'  ~ 'TN',
      loan_status == 'Unpaid' & prediction == 'Paid'  ~ 'FN'))) 
TP <- length(which(results$contingency == 'TP'))
TN <- length(which(results$contingency == 'TN'))
FP <- length(which(results$contingency == 'FP'))
FN <- length(which(results$contingency == 'FN'))
n  <- length(results$contingency)

table(results$contingency) # comprobar recuento de TP, TN, FP, FN

accuracy <- (TP + TN) / n
error <- (FP + FN) / n

precision   <- TP / (TP + FP)
sensitivity <- TP / (TP + FN)
specificity <- TN / (TN + FP)
f_measure   <- (2 * TP) / (2 * TP + FP + FN)
f_measure

## -------------------------------------------------------------------------------------

## 3.Modelo de Prediccion RN
##  Crear modelo de prediccion usando RN
nnCtrl <- trainControl(verboseIter = F, classProbs = TRUE, method = "repeatedcv", 
                       number = 10, repeats = 1, summaryFunction = twoClassSummary)
nnParametersGrid <- expand.grid(.decay = c(0.5, 0.1), .size = c(20, 2000, 2))
nnModel <- train(loan_status ~ ., data = train, method = "nnet", metric = "ROC", 
                 tuneGrid = nnParametersGrid, trControl = nnCtrl, trace = FALSE, maxit = 50000) 
my_roc(val, predict(nnModel, val, type = "prob"), "loan_status", "Unpaid")

# Validacion
prediction     <- predict(nnModel, val, type = "raw")
predictionProb <- predict(nnModel, val, type = "prob")

auc <- roc(val$loan_status, predictionProb[["Unpaid"]], levels = unique(val[["loan_status"]]))
roc_validation <- plot.roc(auc, ylim=c(0,1), type = "S" , print.thres = T, main=paste('[Modelo RN] Validation AUC:', round(auc$auc[[1]], 2)))


# Obtener valores de accuracy, precision, recall, f-score (manualmente)
results <- cbind(val, prediction)
results <- results %>%
  mutate(contingency = as.factor(
    case_when(
      loan_status == 'Unpaid' & prediction == 'Unpaid' ~ 'TP',
      loan_status == 'Paid'  & prediction == 'Unpaid' ~ 'FP',
      loan_status == 'Paid'  & prediction == 'Paid'  ~ 'TN',
      loan_status == 'Unpaid' & prediction == 'Paid'  ~ 'FN'))) 
TP <- length(which(results$contingency == 'TP'))
TN <- length(which(results$contingency == 'TN'))
FP <- length(which(results$contingency == 'FP'))
FN <- length(which(results$contingency == 'FN'))
n  <- length(results$contingency)

table(results$contingency) # comprobar recuento de TP, TN, FP, FN

accuracy <- (TP + TN) / n
error <- (FP + FN) / n

precision   <- TP / (TP + FP)
sensitivity <- TP / (TP + FN)
specificity <- TN / (TN + FP)
f_measure   <- (2 * TP) / (2 * TP + FP + FN)
f_measure
