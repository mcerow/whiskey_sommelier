# Whiskey Sommelier
Bryan Santons, Rajeev Panwar & Maura Cerow

## Introduction

This project aims to build the first module of a comprehensive whisky recommendation engine, an **automated whisky sommelier**. We would like this to act as a go-to "middle-man" between distributors, bars and consumers.

The US whiskey market is expected to grow at a CAGR of 2.4% and the industry faces a lot of competition in the US from other liquor segments. The whiskey market in particular has a low per capita revenues of only $57 per consumer point so there is a lot of room for growth. One of the issues when it comes to entering the whiskey market is that finding the right whiskey for you can be difficult. It can be intimidating to see all the options and not know where to go. With our analysis, the goal is to have an app where a customer can input criteria like price range, certain flavors, user ratings, etc and it will return the options that fit that criteria. Now this is ambitious and we realize that, so our first goal is to prime the underlying data for such an undertaking.

The data used in this project was pulled from https://www.distiller.com.  When we gathered our data, it wasn't ready to be used in such a capacity. It was missing one of the key search functions -- country of origin. Therefore, our target variable in our classifier model is 'country' relating to the country of origin for a given whiskey. The features used include:
  - Type (bourbon, single malt, etc)
  - Price (ranging from 1 to 5 with 5 being the most expensive)
  - Expert Score (critic score)
  - User Rating (distiller.com user reviews)
  - Age
  - ABV (Alcohol By Volume)
  - Description
  - Style (ingredients)
  - Maturing/Cask (whether matured in Oak, Sherry, etc.)
  - Flavor Profiles (each having intensity scores ranging from 0 to 100):
     - Smoky
     -  Peaty
     - Spicy
     - Herbal
     - Oily
     - Full-Bodied
     - Rich
     - Sweet
     - Briny
     - Salty
     - Vanilla
     - Tart
     - Fruity
     - Floral
Some features we've added are - 
        - big_ticket (whiskeys that fall in the higher price point buckets)
        - poor_performance (whiskeys with a low expert score)
        - people_love_this (whiskeys that have a high user rating and are the most expensive)
        - bang for the buck metrics:
            * expert score per price
            * user rating score per price
        - flavor intensity

The questions we're answering in this model - 
  1. Can we confidently classify the country of origin given a set of features?
  2. Do we have an even distribution of data among the 5 countries we're predicting for?
  3. Does price vary depending on the country of origin?
  4. Do expert scores differ between the countries?
  5. Do users prefer one kind of whiskey over another?
  6. Does alcohol levels differ between the countries?
  7. Does the whiskey flavor profile vary from country to country?
    
The following libraries were used for this project: 

### Data Collection
  1. Requests
  2. Beautiful Soup
  3. NumPy
  4. Pandas
  5. urllib
  6. Time
  7. TQDM
  
### Data Cleaning
  1. NumPy
  2. Pandas
  
### EDA & Feature Engineering
  1. NumPy
  2. Pandas
  3. Matplotlib
  4. Seaborn
  5. Missingno
  
### Hypothesis Testing
  1. NumPy
  2. Pandas
  3. Matplotlib
  4. Seaborn
  5. OLS from Statsmodels.formula.api
  6. Statsmodels.api
  7. Pairwise_tukeyhsd from Statsmodels.stats.multicomp
  8. Multicomparison from Statsmodels.stats.multicomp
  9. Scipy.stats
  10. Scipy

### Classification Modeling Baseline & Final Model Evaluation
##### Same libraries were used for both of these notebooks.
  1. NumPy
  2. Pandas
  3. Metrics from sklearn
  4. From sklearn.model_selection:
      - train_test_split
      - cross_val_score
      - GridSearchCV
      - StratifiedKFold
      - learning_curve
  5. From sklearn.metrics:
      - mean_squarred_error
      - r2_score
      - confusion_matrix
      - classification_report
      - f1_score
      - roc_curve
      - roc_auc_curve
  6. LogisticRegression from sklearn.linear_model
  7. From sklearn.preprocessing:
      - StandardScaler
      - binarize
   8. KNeighborsClassifier from sklearn.neighbors
   9. From sklearn.tree:
      - DecisionTreeRegressor
      - DecisionTreeClassifier
      - export_graphviz
  10. From sklearn.ensemble:
      - RandomForestClassifier
      - AdaBoostClassifier
      - GradientBoostingClassifier
  11. LabelBinarizer from sklearn.preprocessing
  12. imblearn
  13. From imblearn.under_sampling:
      - RandomUnderSampler
      - TomekLinks
      - ClusterCentroids
   14. From imblearn.over_sampling:
      - RandomOverSampler
      - SMOTE
   15. SMOTETomek from imblearn.combine
   16. GaussianNB from sklearn.naive_bayes
   17. XGBClassifier from xgboost
   18. StringIO from sklearn.externals.six
   19. Image from IPython.display
   20. pydotplus
   21. pprint
   22. itertools
   23. Statsmodels.api
   24. OLS from statsmodels.formula.api
   25. Matplotlib
   26. Seaborn
   27. Loadtxt from NumPy
   28. Warnings

In this repo, you will find the jupyter notebooks that correspond to the steps we followed when building our classifiers. The original csv for our scraped data is included, but you will find at the bottom of each notebook the code to save down a csv the dataset.

## Data Collection
*See: 01_data_web_scraping_final.ipynb

For this project, we scraped data from distillery.com for the available whiskeys. The first step was to get the links for each whiskey's details. We had to browse through results with 50 results on each page and was then able to get the HREF link for each. From there, each link was appended to a list to then start the scraping process. From our list of links, we were able to parse through to get the pertinent values required for our analysis.

We were able to assemble 2,800 observations and saved as a csv to move on to the next stage.

## Data Cleaning
*See: 02_data_cleaning_final.ipynb

When we got our data from distillery.com, it came through with unwanted characters, ie '\n', surrounding some of the values in our columns. Before proceeeding to EDA and later modeling, we needed to clean our data up. We removed all of the unwanted characters from the 'name' and 'price' column. Because we are setting out to predict the country of origin for a particular whiskey, we needed to determine that from the distillery location. Once we had the country, we assigned numerical values to proceed.

The country codes are:

    0 - Scotland
    1 - America
    2 - Canada
    3 - Ireland
    4 - Japan
    
The final check we wanted to make was around the 'age' column. Many of our values came in as null. Where some weren't null, they didn't have usable. Because of this, we decided to proceed without age for our current analysis.

From here we'll be working with the 'whiskey_df_clean.csv'.

## Hypothesis Testing
*See: 03_hypothesis_testing_final.ipynb

In our hypothesis testing, we look to answer the questions below:

  * Do we have an even distribution of data among the 5 countries we're predicting for?
  * Does price vary depending on the country of origin?
  * Do expert scores differ between the countries?
  * Do users prefer one kind of whiskey over another?
  * Does alcohol levels differ between the countries?
  * Does the whiskey flavor profile vary from country to country?
  
The goal of answering these questions is to justify including these features in our model. By running these hypothesis tests, we are attempting to prove that they have statistically significant differences and are therefore useful in determining our output variable.

## EDA & Feature Engineering
*See: 04_eda_feature_engineering_final.ipynb

Before we model, we want to understand our data. The first thing to check is our target variable. See below for the distribution of our target variable.

![](images/target_distribution.png)

By looking at our graph, we know we have some class imbalance. We'll want to address this when running our model.

Our EDA also includes addressing null values in the user rating columns. We don't want to lose observations here, so instead we will fill these null values by creating a weight which is our average expert score divided by the average user score. With this weight, we multiply the expert score for an observation by the weight to get a suitable user score.

We checked for correlation between the flavor profiles. We'll want to address multicollinearity and these features could potentailly explain one another.

![](images/flavor_correlation.png)

Another feature with null values is cask. We want to make the most of this column since it's instrumental in the whiskey making process. We have the 'style' column that, in some instances, tell us the cask type. We went through and pulled out the casks where we could. When we checked the number of unique values in this column, we found we had 761. Because these values are user input, we had the same value just written different ways for many of them. We created a new column called 'cask_category' to parse through and find the key words to identify the cask. After we were able to get, we created dummies for this categorical variable. We have a catch all 'other' category for any observations that didn't have a specified cask or one off value.

We wanted to check for relationships between our features. The price, expert_score & user_rating were a good place to start. When looking at the breakout of our price data, we saw that American whiskeys are more often in the lower price ranges and Scotch & Japanese whiskey are more expensive. We created the 'big_ticket' column to capture this pattern.

![](images/price_by_country.png)

We checked the relationship between price and expert_score, expecting that as the price went up, so did the expert_score. Overall, American whiskeys tended to have less favorable scores, so we added another new feature 'poor_performance' to track this.

![](images/price_expert.png)

Another indication we wanted to look into was the price & user_rating. One thing that stuck out was that for the most expensive bucket, Irish whiskey was noticeably higher rated. With this insight, we added the 'people_love_this' column. 

![](images/price_user.png)

Our other new features relate to the best bang for our buck - expert_score per price. This gives a qualitative value per price point to check if a whiskey is indeed bang-for-the-buck. The higher it is, the better is is for the money. We also did the same for user_rating per price.

The last feature we added was around the flavor profile. We calculated each whiskey's total flavor intensity by getting the intesity scores' mean of all 14 flavor profiles. It might be fascinating to look whether whiskeys from certain countries have more subtle flavor nuances.

## Classificaion Model Baseline
*See: 05_classification_modeling_baseline_final.ipynb

Just a head's up, this notebook is long!! This notebook include the functions to build a model using different classifiers:

  - Logistic Regression
  - K-Nearest Neighbors
  - Decision Trees
  - Random Forest
  - Adaptive Boost
  - Gradient Boost
  - XGBoost
  - Naive Bayes
  
These are the initial classification models. We did not yet address the class imbalance for these first models. Our next step is then to use different class imbalance solutions with each classifier method. To address class imbalance, we useL

  - Random Undersampling
  - Random Oversampling
  - Tomek Links
  - Cluster Centroids
  - SMOTE
  - SMOTE Tomek Links

For each instance, we compare:

  - Accuracy
  - Precision
  - Recall	
  - F1 Score
  - ROC AUC Score
  
We ran 98 models. That's a lot to have to tune, so we proceeded with the top 5 models based on the F1 Score in the next notebook.

## Model Evaluation
*See: 06_hypertuning_final_model_evaluation_selection_final.ipynb

We saved down a csv with our model metrics from the '05_classification_modeling_baseline_final.ipynb' notebook. We want to tune the top 5 models to pick our best one.

Before picking the top 5, we wanted to capture how each of classifiers predicted our model by looking at the average F1 score. Why did we choose to look at the F1 score? Because in this case we want to weight precision and recall equally. Before tuning the hyperparameters, we lay out what parameters we'll be tuning & why.

Ultimately we land on our best model - Random Forest with Random Oversampling. Our F1 Score for this model is 87.3% and the ROC AUC score is 90.7%. Balanced Precision and Recall with .886 and .884 values respectively.

Our model doesn't get everything right. It's good at detecting Scotland, America and Canada, but not as great as predicting Ireland and Japan. They are both classified incorrectly as Scotland the most indicating that based on our current feautures, it is difficult to delineate between these two classes and Scotland. One way we can adjust for this is introducing new features to our dataset. We know that color, aroma and finish are additional features in whiskey that could potentially help classify them.

It is interesting that these top models had different top features in terms of importance. The winning models had the flavor profile: Oily as the top but the other models had one of the categorical cask variables and one of the new features engineered - flavor intesity. This shows the our feature engineering was effective to say the least.

## Conclusion & Future Steps

So we have this amazing model and all this data surrounding whiskey. Our key takeaways are:

  * Whiskey differs from country to country. Our final model is able to classify whiskey types with an accuracy score of 88.4%     and an F1 score of 86.8%. Our best model is a Random Forest using Random Oversampling to adjust for class imbalance. 
  * Our categories are in fact imbalanced. Because of this, we used SMOTE, TomekLink and other imbalance solutions to correct     the issue. When we originally ran our model without handling the imbalance, our model proved a poor predictor for Canada,
    Ireland and Japan.
  * Whiskey prices do in fact vary depending on what country they come from and price is a good indication for whiskey.
  * Country expert scores are significantly different one another and it is a useful tool for our classificaiton model.
  * User ratings also vary when comparing one country to another.
  * ABV percentages are another indication for country.
  * Finally, flavor profile differs among the different countries. The flavor profile 'oily' is actually the most significant     feature in our final model.

Our next steps regarding our automated whiskey sommelier are:

  * Mapping the population and including whiskeys that fall outside the 5 we have so far designated
  * Adding new features to our model - color, nose & finish
  * Create a NLP classifier to bucekt customer reviews
  * Refine our 'x' factor
  * Build prototype GUI & begin A/B testing 

## Presentation

[whiskey_sommelier](https://docs.google.com/presentation/d/1ENZpW1YLgLV1PvHDEQKhzcgFzJlfmcALtxzVDWTN5C4/edit#slide=id.g73a2f809fc_2_214)
