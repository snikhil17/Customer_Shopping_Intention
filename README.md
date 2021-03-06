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

## Appraoch - Notebooks: 
### Stratify_5_Fold_training_data:
- Created 5-Stratified Folds using StratifiedKFold from sklearn (Data is unbalanced)
- New csv file is created with additional column called ``kfold`` and this will be used for creating validation set.

### EDA_Feature_importance_selection:
- Exploratory Data Analysis
- Feature Selection techniques using sklearn
- Modelling and evaluation of models created using selected features

### Hyper-parameter_tuning_model-building:
- **Hypertuning using Optuna:** 
  - I've selected f1-score as metrics to decide the final paramters because our data is unbalanced.
  - For this pupose i've split the data into 5 Stratified Folds which is shown in a separate notebook.
  - Next I've calculated f1-score for both training and validation set.
  - For the final function i'm returning the mean_difference of these f1-scores i.e. mean(training_f1) - mean(validation_f1).
  - This actually returns a value and we need this distance to be as minimum as possible. Why? because if we observe then validation_f1 must be greater and thats possible when the value is as small as possible (negatively)
- **Model Building:**
  - Plan here is to build 4 different models each optimized for hyper-parameters using Optuna.
  - Finally we use these models with Voting Classifier and get the results using all 4 models.
- **Deployement code**


# How to use.
- Clone the Repository
- Create a virtual environment (recommended but not compulsory) 
- Make sure you are in the Repository folder. 
- Run ``pipenv install`` to install all packages as per the dependencies required for this project. 
- Use ``pipenv run predict.py`` to run the app locally and then can check the prediction on the given dictionary of variables and values using ``pipenv run predict_test.py``

# Docker:

## To use the already created image and run on local machine:

#### **Download the image running following command on anaconda promt or terminal (MAC or Linux)**
- ``docker pull snikhil17/customer_shopping_intention``
#### **Use following command on anaconda promt or terminal (MAC or Linux) to run the image locally i.e. to run the application (we are using Flask and waitress to serve the app).**
- ``docker run -it --rm -p 9696:9696 snikhil17/customer_shopping_intention``
#### **Open another anaconda promt or terminal (MAC or Linux) and navigate to the same location as above and then use following command to get the results as per the variables specified in ``predict_test.py``**
- ``python predict_test.py``

### If you choose to build a docker file locally instead, here are the steps to do so:
- This allows us to install python, run pipenv and its dependencies, run our predict script and our model itself and deploys our model using Flask/gunicorn.

- Create a Dockerfile as such:
  - ``FROM python:3.8.12-slim``
  - ``RUN pip --no-cache-dir install pipenv``
  - ``WORKDIR /app``
  - ``COPY . /app``
  - ``RUN pipenv install --deploy --system``
  - ``EXPOSE 9696``
  - ``ENTRYPOINT ["waitress-serve", "--listen=0.0.0.0:9696", "predict:app"]``

### Similarly, you can just use the dockerfile in this repository.
- Build the Docker Container with :
  - ``docker build -t customer_shopping_prediction .``
- Run the Docker container with:
  - ``Docker run -it -p 9696:9696 customer_shopping_prediction``
- Now we can use our model through
  - ``python predict_test.py``
  
## Virtual environment and package dependencies
- To ensure all scripts work fine and libraries used during development are the ones which you use during your deployment/testing, Python venv has been used to manage virtual environment and package dependencies. Follow the below steps to setup this up in your environment.
- The steps to install Python venv will depend on the Operating system you have. Below steps have been provided in reference to installation on Ubuntu, however you can refer to Official documentation at https://docs.python.org/3/tutorial/venv.html to know appropriate steps for your OS.
- Install pip3 and venv if not installed (below sample commands to be used on Ubuntu hav been provided
  - ``sudo apt install -y python3-pip python3-venv``
- Create a virtual environment. Below command creates a virtual environment named mlzoomcamp
  - ``python3 -m venv mlzoomcamp``
- Activate the virtual environment.
  - ``. ./mlzoomcamp/bin/activate``
- Clone this repo
  - ``git clone https://github.com/snikhil17/Mid-Term-Project-Zoomcamp.git``
- Change to the directory that has the required files
  - ``cd mlzoomcamp-midterm-project/``
- Install packages required for this project
  - ``pip install -r requirements.txt``
