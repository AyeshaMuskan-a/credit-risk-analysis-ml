-- credit-risk-analysis

-- Create Database
CREATE DATABASE credit_risk_db;

-- Use Database
USE credit_risk_db;

-- View all data
SELECT * FROM credit_risk_dataset;

-- View First 10 Rows
SELECT * FROM credit_risk_dataset LIMIT 10;

-- Count Total Records
 SELECT COUNT(*) AS total_customers FROM credit_risk_dataset;
 
-- Average Income
SELECT AVG(annual_income) AS avg_income FROM credit_risk_dataset;

-- Maximum Credit Score
SELECT MAX(credit_score) AS max_score FROM credit_risk_dataset;

-- Minimum Credit Score
SELECT MIN(credit_score) AS min_score FROM credit_risk_dataset;

-- Count by Credit Risk
SELECT credit_risk, COUNT(*) 
FROM credit_risk_dataset
GROUP BY credit_risk;

-- Average Income by Risk
SELECT credit_risk, AVG(annual_income) 
FROM credit_risk_dataset
GROUP BY credit_risk;

-- Average Loan Amount
SELECT AVG(loan_application_amount) FROM credit_risk_dataset;

-- Customers with High Debt
SELECT * FROM credit_risk_dataset
WHERE total_outstanding_debt > 5000;

-- Customers with Good Credit Score
SELECT * FROM credit_risk_dataset
WHERE credit_score > 700;

-- Employment-wise Loan Amount
SELECT employment_status, AVG(loan_application_amount)
FROM credit_risk_dataset
GROUP BY employment_status;

-- High Risk Customers
SELECT * FROM credit_risk_dataset
WHERE credit_risk = 1;

-- Late Payment Analysis
SELECT AVG(late_payment_count) 
FROM credit_risk_dataset
WHERE credit_risk = 1;

-- Fraud vs Risk
SELECT fraud_flag, COUNT(*) 
FROM credit_risk_dataset
GROUP BY fraud_flag;

-- Window Function (Ranking by Income)
SELECT customer_id, annual_income,
RANK() OVER (ORDER BY annual_income DESC) AS income_rank
FROM credit_risk_dataset;

-- Top 5 High Income Customers
SELECT * FROM credit_risk_dataset
ORDER BY annual_income DESC
LIMIT 5;

-- Running Total (Window Function)
SELECT customer_id, annual_income,
SUM(annual_income) OVER (ORDER BY annual_income) AS running_total
FROM credit_risk_dataset;

-- Default Rate/ RISK PERCENTAGE
SELECT (SUM(credit_risk) * 100.0 / COUNT(*)) AS default_rate_percentage
FROM credit_risk_dataset;

-- Top Risky Customers (High Debt + Low Score)
SELECT * 
FROM credit_risk_dataset
WHERE credit_score < 600 
AND total_outstanding_debt > 5000;

-- filter customers earning above average income
SELECT *
FROM credit_risk_dataset
WHERE annual_income > (
    SELECT AVG(annual_income) FROM credit_risk_dataset
);

-- Average Credit Score by Employment
SELECT employment_status, AVG(credit_score) AS avg_score
FROM credit_risk_dataset
GROUP BY employment_status;

-- Shows category understanding of employment_status
SELECT DISTINCT employment_status FROM credit_risk_dataset;

