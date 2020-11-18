# West Nile Virus Prediction
---

## Executive Summary
---
According to the Center for Disease Control and Prevention (CDC), the West Nile virus (WNV) is the leading mosquito-borne disease in the United States. It is a single-stranded RNA virus from the family Flaviviridae, which also contains the Zika, dengue, and yellow fever virus. It is mainly transmitted to humans through the bite of an infected mosquito. Cases of WNV are mainly observed in the United States during mosquito seasons, which starts in the summer and continues through fall (May to October). 


Most human infections are mild (causing fever, headache, and body aches, often accompanied by a skin rash and swollen lymph glands), however, it is known that if the virus crosses the blood-brain barrier, it can cause life-threatening conditions that include inflammation of the brain and spinal cord. 
Although the first case of WNV first emerged in the United States in 1999, there is still no vaccine to prevent or medication to treat the WNV. 

### Problem Statement
---
Currently, local governments and mosquito control programs use an integrated mosquito management (IMM) or integrated vector management (IVM) approach to control mosquitoes. Part of the IVM involves periodic spraying of adulticides to curb the number of mosquitoes breeding, which brings down the number of mosquitoes and in turn, the infected mosquitoes which spread the WNV.

However, there are still cases of WNV reported today (especially in seasonal months) which calls for a need to pinpoint the location and time of spread to identify susceptible areas and improve the IVM.

As part of the Disease And Treatment Agency division of Societal Cures In Epidemiology and New Creative Engineering (DATA-SCIENCE), we were tasked by the Chicago Department of Public Health (CDPH) to set up a surveillance and control system, and identify the area and time of spread, to aid the CDPH in making well-informed and cost-effective decisions in the allocation of resources to such vulnerable locations.

With the development of an Extra Trees classifier model, we intend to predict the time and location susceptible to WNV infections. This model was evaluated through the AUC metric, which optimises the threshold between the true positives and true negatives. 




### Methodology
---
#### Import Data
1) train.csv (10,506 rows) from 2007-05-29 to 2013-09-26<br>
2) weather.csv (2,944 rows) from 2007-05-01 to 2014-10-31<br>
3) spray.csv (14,835 rows) from 2011-08-29 to 2013-09-05<br>
4) test.csv (116,293 rows) from 2008-06-11 to 2014-10-02<br>
#### Data Cleaning
1) Removal of duplicates<br>
2) Implementing imputation <br>
3) Removal of correlated features <br>
#### Feature Engineering
1) Data augmentation: Synthetic Minority Over-Sampling Technique SMOTE(method to oversample minority class)
2) Year, Month, Week of year columns
3) Night Hours (Mosquitoes activity could be higher when night hours are longer)
4) Humidity (Mosquitoes tend to thrive in humid environment)
5) Rolling averages on selected weather features on temperature and precipitation (effect on mosquito embryonation process)
#### Preprocessing
1) Drop redundant columns (Date, Latitude, Longitude, Block, Street, Address and some weather events etc.)
2) Classify certain less significant mosquito species as “other species”
3) Group traps according to the number of wnvpresent
3) Apply one hot encoding
#### Modelling
1) Apply Train-test-split on train dataset
2) Apply Standard Scaler on X_train and transform X_test
3) Fit train set into Pipeline consisting of SMOTE and each classification model
4) Tune hyperparameters using GridSearchCV
5) Generate Predictions
#### Cost-Benefit Analysis
1) Find out if sprays are effective in controlling number of mosquitoes
2) Weigh costs and benefits associated with using sprays to combat WNV infection rates 


### Data Dictionary
---

|Feature|Type|Dataset|Description|
|:---|:---|:---|:---|
|Id            |int      |train.csv/test.csv |ID number of the record
|Date           |datetime      |train.csv/test.csv |date the WNV test is performed
|Address      |datetime |train.csv/test.csv| approximate trap address; sent to GeoCoder
|Species         |str      |train.csv/test.csv| mosquito species in trap
|Block         |str      |train.csv/test.csv| block number of address
|Street        |str      |train.csv/test.csv| street of address
|Trap          |str    |train.csv/test.csv| ID number of the trap
|AddressNumberAndStreet|str|train.csv/test.csv| approximate address retrieved from GeoCoder
|Latitude            |float|train.csv/test.csv| latitude retrieved from GeoCoder
|Longitude            |float|train.csv/test.csv| longitude retrieved from GeoCoder
|AddressAccuracy      |int|train.csv/test.csv| accuracy of information returned from GeoCoder
|NumMosquitos        |int      |train.csv/test.csv| number of mosquitoes in a sample
|WnvPresent           |int      |train.csv/test.csv| whether or not WNV is present in a sample (1 = present; 0 = absent)
|Date |datetime |spray.csv| date of spray
|Time         |datetime      |spray.csv| time of spray
|Latitude|float|spray.csv| latitude of spray
|Longitude        |float|spray.csv| longitude of spray
|Station  |int|weather.csv| weather station (1 or 2)
|Date     |datetime|weather.csv| date of measurement
|Tmax     |int|weather.csv| maximum daily temperature (F)
|Tmin     |int|weather.csv| minimum daily temperature (F)
|Tavg  |int|weather.csv| average daily temperature (F)
|Depart     |int|weather.csv| departure from normal temperature (F)
|Dewpoint    |int|weather.csv| average dewpoint (F)
|WetBulb  |int|weather.csv| average wet bulb
|Heat     |int|weather.csv| heating degree days
|Cool  |int|weather.csv| cooling degree days
|Sunrise     |int|weather.csv| time of sunrise (calculated)
|Sunset     |int|weather.csv| time of sunset (calculated)
|CodeSum     |str|weather.csv| code of weather phenomena 
|Depth     |int|weather.csv| unknown
|Water1  |int|weather.csv| unknown
|SnowFall     |int|weather.csv| snowfall (inch)
|PrecipTotal     |str|intweather.csv| total daily rainfall (inch)
|StnPressure     |int|weather.csv| average atmospheric pressure (inch Hg)
|SeaLevel     |int|weather.csv| average sea level pressure (inch Hg)
|ResultSpeed  |float|weather.csv| resultant wind speed (mph)
|ResultDir     |int|weather.csv| resultant wind direction (degrees)
|AvgSpeed     |int|weather.csv| average wind speed (mph)


### Findings
---

During our initial Exploratory Data Analysis (EDA), `Number of Mosquitoes` is found to be the top predictor of the model, followed by `Week of year` and a `14 day rolling average of relative humidity`. Across the years of our dataset, WNV is observed to be most prevalent in the 3rd week of the month of August. 


#### Best Model
<u>Models trained:</u><br>
- Logistic Regression
- K Nearest Neighbours, 
- Decision Tree Classifier 
- Bagging Classifier 
- Random Forest Classifier 
- Extra Trees Classifier 
- AdaBoost Classifier 
- XGBoost Classifier

Through training of 8 different classification models, we have selected the Extra Trees Classifier as our production model. 

|Model| AUC score | Recall |
|:---|:---|:---|
|ExtraTreeClassifier |0.83|0.73|


Based on our production model, months of the year, duration of night hours, mosquito species (Culex Restuans and Culex Pipiens) and location (traps without WNV presence and T900 at Ohare Airport) make up the top predictors of the presence of WNV in the traps. Similar to our initial analysis during the EDA, it seems that the month of August is one of the most important features in making our predictions. Week of year and the 14 day rolling average of relative humidity remained to be amongst the top 20 features in our production model.


#### Cost Benefit Analysis
| Type     | Cost                         |
|----------|------------------------------|
| Medical  | \$840 000 at a minimum |
| Spraying | \$800 000                     |


This seems to be an obvious choice to choose one of the vector control methods, spraying of insecticide to save future costs impacted by the existence of the West Nile Virus. 

However, spraying is not the only measure that can be done on the public health side. Frequent announcements of which areas are potential hotspots also serve to educate the public to take more preventive action on their own. We also have not factored in the hidden costs of spraying, but we can be sure that in the short to medium term, spraying during the months of July and August have a net postive effect on public health in Chicago

### Conclusion & Recommendations

In order to best utilise the resources in the most cost effective way, we recommend the following:
1) Concentrate spraying efforts in the month of July, since the most of the outbreaks happen in August. It seems that the mosquitoes start breeding and multiplies exponentially during July which can affect the infection rates.
Another thing to note is the incubation period of the virus which is approximately 14 days, which meant that the mosquitoes could have been infecting humans, in July which effects the peak of virus outbreaks at August.
2) Spray periodically (once a week) during seasonal months (up to end of September) to keep mosquito numbers less than 2000 per trap. 
3) Concentrate spraying efforts to vulnerable locations as identified in the map below:
Based on our production model, it is predicted that these locations are susceptible to the spread of the WNV. 

<img src="\assets\image2008.jpg">
<img src="\assets\image2010.jpg">
<img src="\assets\image2012.jpg">
<img src="\assets\image2014.jpg">
