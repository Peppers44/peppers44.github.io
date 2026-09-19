library(caret)
library(rpart)

data = read.csv("loan_data.csv", na.strings = c("NA", "Unknown", "", " "))

data$ApplicantIncome[is.na(data$ApplicantIncome)] = median(data$ApplicantIncome, na.rm = TRUE)
data$CoapplicantIncome[is.na(data$CoapplicantIncome)] = median(data$CoapplicantIncome, na.rm = TRUE)

Q1a = quantile(data$ApplicantIncome, 0.25)
Q3a = quantile(data$ApplicantIncome, 0.75)
IQRa = IQR(data$ApplicantIncome)

Q1c = quantile(data$CoapplicantIncome, 0.25)
Q3c = quantile(data$CoapplicantIncome, 0.75)
IQRc = IQR(data$CoapplicantIncome)

cat("ApplicantIncome Outliers:", sum(data$ApplicantIncome < (Q1a - 1.5*IQRa) | data$ApplicantIncome > (Q3a + 1.5*IQRa)), "\n")
cat("CoapplicantIncome Outliers:", sum(data$CoapplicantIncome < (Q1c - 1.5*IQRc) | data$CoapplicantIncome > (Q3c + 1.5*IQRc)), "\n")

cleaned_data = subset(data, 
ApplicantIncome >= (Q1a - 1.5*IQRa) & ApplicantIncome <= (Q3a + 1.5*IQRa) &
CoapplicantIncome >= (Q1c - 1.5*IQRc) & CoapplicantIncome <= (Q3c + 1.5*IQRc))

write.csv(cleaned_data, "cleaneddatar.csv", row.names = FALSE)

accepted_loans = subset(cleaned_data, Loan_Status == "Y")
rejected_loans = subset(cleaned_data, Loan_Status == "N")

barplot(table(accepted_loans$Married), main="Figure 1: Married Status vs. Accepted Loans", 
col=c("blue", "orange"), ylab="Count", xlab="Married")

barplot(table(rejected_loans$Married), main="Figure 2: Married Status for Rejected Loans", 
col=c("blue", "orange"), ylab="Count", xlab="Married")

plot(cleaned_data$ApplicantIncome, cleaned_data$CoapplicantIncome, 
main="Figure 3: Applicant vs Coapplicant Income", 
xlab="Applicant Income", ylab="Coapplicant Income", col="darkgreen", pch=16)

par(mfrow=c(1,3))
boxplot(cleaned_data$ApplicantIncome, main="Applicant Income", col="lightblue")
boxplot(cleaned_data$CoapplicantIncome, main="Coapplicant Income", col="lightgreen")
boxplot(cleaned_data$LoanAmount, main="Loan Amount", col="salmon")
par(mfrow=c(1,1)) 

fivenum(cleaned_data$ApplicantIncome)
fivenum(cleaned_data$CoapplicantIncome)
fivenum(cleaned_data$LoanAmount)

cleaned_data = cleaned_data[, !(names(cleaned_data) %in% c("Loan_ID", "Applicant_ID", "Owns_Car"))]
cleaned_data[sapply(cleaned_data, is.character)] <- lapply(cleaned_data[sapply(cleaned_data, is.character)], as.factor)
cleaned_data = na.omit(cleaned_data)

set.seed(42)
trainIndex = createDataPartition(cleaned_data$Loan_Status, p = 0.7, list = FALSE)
trainData = cleaned_data[trainIndex, ]
testData  = cleaned_data[-trainIndex, ]

model_dt = rpart(Loan_Status ~ ., data = trainData, method = "class")
preds_dt = predict(model_dt, testData, type = "class")
accuracy_dt = sum(preds_dt == testData$Loan_Status) / nrow(testData)
cat("Acc", accuracy_dt, "cool\n") # ~76.0% Accuracy

set.seed(42)
model_knn = train(Loan_Status ~ ., 
data = trainData, 
method = "knn", 
preProcess = c("center", "scale"), 
tuneGrid = data.frame(k = 5))

preds_knn = predict(model_knn, testData)
accuracy_knn = sum(preds_knn == testData$Loan_Status) / nrow(testData)
cat("KAcc", accuracy_knn, "\n") # ~73.9% Accuracy