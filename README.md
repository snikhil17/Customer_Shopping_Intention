## **Business Problem:**
- The recent surge in online shopping has given the business world a new dimension. People frequently search the internet for the products they require and purchase them through online transactions. 
- It has made their lives more convenient and comfortable. 
- Simultaneously, the sellers must understand the patterns and intentions of the customers.

## **Objective**
- Objective here is to build a Machine Learning model that can help in predicting whether a customer will purchase or not.

## **Data Dictionary:**
### **The dataset consists of 10 numerical and 8 categorical attributes.** 
- The **Revenue** attribute is used as the target label. 
- **Administrative**, **Administrative Duration**, **Informational**, **Informational Duration**, **Product Related** and **Product Related Duration** represent the number of different types of pages visited by the visitor in that session and total time spent in each of these page categories. The values of these features are derived from the URL information of the pages visited by the user and updated in real time when a user takes an action, e.g. moving from one page to another. 
- The **Bounce Rate**, **Exit Rate** and **Page Value** features represent the metrics measured by **Google Analytics** for each page in the e-commerce site. 
- The value of **Bounce Rate** feature for a web page refers to the percentage of visitors who enter the site from that page and then leave (**bounce**) without triggering any other requests to the analytics server during that session. 
- The value of **Exit Rate** feature for a specific web page is calculated as for all pageviews to the page, the percentage that were the last in the session. 
- The **Page Value** feature represents the average value for a web page that a user visited before completing an e-commerce transaction. 
- The **Special Day** feature indicates the closeness of the site visiting time to a specific special day (e.g. Mother’s Day, Valentine's Day) in which the sessions are more likely to be finalized with transaction. The value of this attribute is determined by considering the dynamics of e-commerce such as the duration between the order date and delivery date. For example, for Valentina’s day, this value takes a nonzero value between February 2 and February 12, zero before and after this date unless it is close to another special day, and its maximum value of 1 on February 8. 
- The dataset also includes operating system, browser, region, traffic type, visitor type as returning or new visitor, a Boolean value indicating whether the date of the visit is weekend, and month of the year.

## Notebooks: 
### EDA_Feature_importance_selection:
- Exploratory Data Analysis
- Feature Selection techniques using sklearn
- Modelling and evaluation of models created using selected features

### hyper-parameter_tuning_model-building:
- Hypertuning using Optuna.
- Model building (4 models + 1 Voting Classifier)
- Deployement code
