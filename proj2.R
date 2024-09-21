setwd("C:/Users/hassn/Downloads/house-prices-advanced-regression-techniques") 

##packages
installed.packages("corrplot") 
installed.packages("tidyverse")
installed.packages("caret") 
installed.packages("dplyr") 
installed.packages("tidyr")  
installed.packages("Metrics") 
installed.packages("randomForest")  
installed.packages("glmnet")  
installed.packages("xgboost")  
install.packages("DescTools") 
install.packages('e1071') 
install.packages("ggplot2")
library(e1071)
#install.packages("dummies") 
#install.packages("superml")
#library(superml)
#library(dummy)
#library(dummies)
library(corrplot) 
library(tidyverse) 
library(caret)  
library(dplyr) 
library(tidyr) 
library(Metrics) 
library(randomForest)
library(glmnet)
library(xgboost)
library(DescTools) 
library(ggplot2) 

# Load data
train_df <- read.csv("train.csv") 


#Remove columns with more than 80% missing values
train_df<-subset(train_df,select = -c(Id,Alley,PoolQC,Fence,MiscFeature,FireplaceQu))
train_df <- train_df[, colMeans(is.na(train_df)) < 0.8] 
str(train_df)

#Fill missing values
train_df$MasVnrType[is.na(train_df$MasVnrType)] <- "None"
train_df$MasVnrArea[is.na(train_df$MasVnrArea)]<-median(train_df$MasVnrArea,na.rm=TRUE)
train_df$BsmtQual[is.na(train_df$BsmtQual)] <- "None"
train_df$BsmtCond[is.na(train_df$BsmtCond)] <- "None"
train_df$BsmtExposure[is.na(train_df$BsmtExposure)] <- "None"
train_df$BsmtFinType1[is.na(train_df$BsmtFinType1)] <- "None"
train_df$BsmtFinType2[is.na(train_df$BsmtFinType2)] <- "None"
train_df$Electrical[is.na(train_df$Electrical)] <- "SBrkr"
train_df$GarageType[is.na(train_df$GarageType)] <- "None"
train_df$GarageYrBlt[is.na(train_df$GarageYrBlt)] <- train_df$YearBuilt[is.na(train_df$GarageYrBlt)]
train_df$GarageFinish[is.na(train_df$GarageFinish)] <- "None"
train_df$GarageQual[is.na(train_df$GarageQual)] <- "None"
train_df$GarageCond[is.na(train_df$GarageCond)] <- "None" 
train_df$LotFrontage[is.na(train_df$LotFrontage)]<-median(train_df$LotFrontage,na.rm=TRUE)
#colSums(is.na(train_df))

# Encode categorical variables
categorical_cols <- sapply(train_df, function(x) is.factor(x) || is.character(x))
train_df[categorical_cols] <- lapply(train_df[categorical_cols], as.factor)
train_df[categorical_cols] <- lapply(train_df[categorical_cols], function(x) as.numeric(x)) 
#print(train_df)
dim(train_df)

# removing outliers 
# winsorizing  method 
numeric_cols <- sapply(train_df, is.numeric)
#train_df[numeric_cols] <- lapply(train_df[numeric_cols], function(x) Winsorize(x, probs = c(0.01, 0.99)))

#interquartile range method 
handle_outliers <- function(variable) {
  Q1 <- quantile(variable, 0.25)
  Q3 <- quantile(variable, 0.75)
  IQR <- Q3 - Q1
  lower_bound <- Q1 - 1.5 * IQR
  upper_bound <- Q3 + 1.5 * IQR
  variable[variable < lower_bound] <- lower_bound
  variable[variable > upper_bound] <- upper_bound
  return(variable)
}
for (var in numeric_cols) {
  train_df[[var]] <- handle_outliers(train_df[[var]])
} 

#z-score method for outliers 

#remove_outliers_zscore <- function(variable, threshold = 3) {
#  z_scores <- abs((variable - mean(variable)) / sd(variable))
 # variable[z_scores > threshold] <- NA
  #return(variable)
#}
#for (var in numeric_cols) {
 # train_df[[var]] <- remove_outliers_zscore(train_df[[var]])
#}




#Remove columns with low correlation to SalePrice
#cor_matrix <- cor(train_df[, -1]) 
#colSums(is.na(train_df))
#low_corr_cols <- colnames(cor_matrix)[cor_matrix[,"SalePrice"] < 0.3]
#train_df <- train_df[, !colnames(train_df) %in% low_corr_cols]
dim(train_df)  


# Split into train and validation sets
set.seed(123)
train_idx <- sample(nrow(train_df), 0.7 * nrow(train_df))
train_set <- train_df[train_idx, ]
train_set_saleprice <- train_set$SalePrice 
train_set <- train_set[-75]
val_set <- train_df[-train_idx, ] 
val_set_saleprice <- val_set$SalePrice
val_set <- val_set[-75]

#Fit model on train set and predict on validation set
model <- lm(train_set_saleprice ~ ., data = train_set)
val_preds <- predict(model, newdata = val_set)


# Calculate MSE and RMSE
mse <- mean((val_preds - val_set_saleprice)^2)
rmse <- sqrt(mse)
print(paste0("MSE: ", mse))
print(paste0("RMSE: ", rmse))
mae <- MAE(val_preds, val_set_saleprice) 
print(mae)  

#Random Forest model
rf_model <- randomForest(y =train_set_saleprice , x = train_set, ntree = 500)

#Lasso model
#lambda_seq <- 10^seq(10, -2, length = 100)
#lasso_model <- cv.glmnet(x = train_set, y = train_set_saleprice, alpha = 1, lambda = lambda_seq)

# XGBoost model
xgb_model <- xgboost(data = as.matrix(train_set), label = train_set_saleprice, nrounds = 100, objective = "reg:linear", eta = 0.1, max_depth = 3)

# Predict on validation set using each model
rf_preds <- predict(rf_model, newdata = val_set)
xgb_preds <- predict(xgb_model, newdata = as.matrix(val_set))

# Calculate MSE and RMSE for each model
rf_mse <- mse(val_set_saleprice, rf_preds)
rf_rmse <- rmse(val_set_saleprice, rf_preds)

xgb_mse <- mse(val_set_saleprice, xgb_preds)
xgb_rmse <- rmse(val_set_saleprice, xgb_preds) 


#Print the MSE and RMSE for each model
cat("Random Forest MSE:", rf_mse, "\n")
cat("Random Forest RMSE:", rf_rmse, "\n")

cat("XGBoost MSE:", xgb_mse, "\n")
cat("XGBoost RMSE:", xgb_rmse, "\n") 





##preprocessing and predicting for test data 
test_df <- read.csv("test.csv") 
#save id
testid<-testdata['Id']

#Remove columns with more than 80% missing values
test_df<-subset(test_df,select = -c(Id,Alley,PoolQC,Fence,MiscFeature,FireplaceQu))
test_df <- test_df[, colMeans(is.na(test_df)) < 0.8] 
#str(test_df)

#Fill missing values
test_df$MasVnrType[is.na(test_df$MasVnrType)] <- "None"
test_df$MasVnrArea[is.na(test_df$MasVnrArea)]<-median(test_df$MasVnrArea,na.rm=TRUE)
test_df$BsmtQual[is.na(test_df$BsmtQual)] <- "None"
test_df$BsmtCond[is.na(test_df$BsmtCond)] <- "None"
test_df$BsmtExposure[is.na(test_df$BsmtExposure)] <- "None"
test_df$BsmtFinType1[is.na(test_df$BsmtFinType1)] <- "None"
test_df$BsmtFinType2[is.na(test_df$BsmtFinType2)] <- "None"
test_df$Electrical[is.na(test_df$Electrical)] <- "SBrkr"
test_df$GarageType[is.na(test_df$GarageType)] <- "None"
test_df$GarageYrBlt[is.na(test_df$GarageYrBlt)] <- test_df$YearBuilt[is.na(test_df$GarageYrBlt)]
test_df$GarageFinish[is.na(test_df$GarageFinish)] <- "None"
test_df$GarageQual[is.na(test_df$GarageQual)] <- "None"
test_df$GarageCond[is.na(test_df$GarageCond)] <- "None" 
test_df$LotFrontage[is.na(test_df$LotFrontage)]<-median(test_df$LotFrontage,na.rm=TRUE)
test_df$BsmtFinSF1[is.na(test_df$BsmtFinSF1)]<-mode(test_df$BsmtFinSF1)
test_df$TotalBsmtSF[is.na(test_df$TotalBsmtSF)]<-mode(test_df$TotalBsmtSF)
test_df$GarageCars[is.na(test_df$GarageCars)]<-mode(test_df$GarageCars)
test_df$GarageArea[is.na(test_df$GarageArea)]<-mode(test_df$GarageArea) 
test_df$MSZoning[is.na(test_df$MSZoning)] <- "None" 
test_df$Utilities[is.na(test_df$Utilities)] <- "None"
test_df$Exterior1st[is.na(test_df$Exterior1st)]<-"None" 
test_df$Exterior2nd[is.na(test_df$Exterior2nd)]<-"None" 
test_df$BsmtFinSF2[is.na(test_df$BsmtFinSF2)]<-"None" 
test_df$BsmtUnfSF[is.na(test_df$BsmtUnfSF)]<-"None"
test_df$BsmtFullBath[is.na(test_df$BsmtFullBath)]<-"None" 
test_df$BsmtHalfBath[is.na(test_df$BsmtHalfBath)]<-"None" 
test_df$Functional[is.na(test_df$Functional)] <-"None" 
test_df$SaleType[is.na(test_df$SaleType)] <-"None"
test_df$KitchenQual[is.na(test_df$KitchenQual)] <- "None"


# Encode categorical variables
categorical_cols <- sapply(test_df, function(x) is.factor(x) || is.character(x))
test_df[categorical_cols] <- lapply(test_df[categorical_cols], as.factor)
test_df[categorical_cols] <- lapply(test_df[categorical_cols], function(x) as.numeric(x)) 

#print(train_df)

# predict on test set
SalePrice <- predict(model, newdata = test_df) 
dt<-cbind(testid,SalePrice)
write.csv(dt,"C:/Users/hassn/Downloads/house-prices-advanced-regression-techniques/linearReg_Submission.csv", row.names = FALSE)

rf_preds <- predict(rf_model, newdata = test_df)
SalePrice<-rf_preds
dt2<-cbind(testid,SalePrice) 
write.csv(dt2,"C:/Users/hassn/Downloads/house-prices-advanced-regression-techniques/RanForest_Submission.csv", row.names = FALSE)

# Visualize the distribution of the target variable (SalePrice)
ggplot(train_df, aes(x = SalePrice)) +
  geom_histogram(fill = "steelblue", color = "white") +
  labs(x = "SalePrice", y = "Count") +
  ggtitle("Distribution of SalePrice")

# Visualize the correlation between numeric features and the target variable
numeric_cols <- sapply(train_df, is.numeric)
numeric_df <- train_df[, numeric_cols]

correlation <- cor(numeric_df)
cor_df <- reshape2::melt(correlation, varnames = c("Variable 1", "Variable 2"), value.name = "Correlation")

ggplot(cor_df, aes(x = `Variable 1`, y = `Variable 2`, fill = Correlation)) +
  geom_tile() +
  scale_fill_gradient2(low = "blue", mid = "white", high = "red", midpoint = 0) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  ggtitle("Correlation between Numeric Features and SalePrice")

# Visualize model predictions vs. actual values
prediction_df <- data.frame(Actual = val_set_saleprice, Predicted = val_preds)

ggplot(prediction_df, aes(x = Actual, y = Predicted)) +
  geom_point(color = "steelblue") +
  geom_abline(intercept = 0, slope = 1, color = "red", linetype = "dashed") +
  labs(x = "Actual SalePrice", y = "Predicted SalePrice") +
  ggtitle("Actual vs. Predicted SalePrice")



#data frame to store model names and corresponding RMSE
model_names <- c("Linear Regression", "Random Forest", "XGBoost")
rmse_values <- c(29000.5447352243, 25609.08 , 23539.43) 

rmse_df <- data.frame(Model = model_names, RMSE = rmse_values)

#Visualize RMSE of different models
ggplot(rmse_df, aes(x = Model, y = RMSE)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  labs(x = "Model", y = "RMSE") +
  ggtitle("RMSE of Different Models") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

