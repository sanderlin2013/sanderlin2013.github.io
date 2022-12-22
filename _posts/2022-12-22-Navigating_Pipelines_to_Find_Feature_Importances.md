# Navigating Pipelines to Find Feature Importances

## Introduction
In my most recent [project](https://github.com/sanderlin2013/Predicting-Flu-Vaccines), I analyzed the CDC’s [National 2009 H1N1 Flu Survey](https://www.drivendata.org/competitions/66/flu-shot-learning/page/211/). The goal of this analysis was to create models that could find the most important features in the survey, so we could use those features to create a new survey on COVID-19 vaccine compliance. The original data included questions about both H1N1 and the seasonal flu vaccine compliance, but I choose to focus on only the seasonal flu. I figured that as COVID-19 vaccines and boosters become more routine like the seasonal flu vaccine, the season flu questions would be more relevant to the new survey. 

I wanted to be able to build and test a few different models, so I could choose the best one to base my analysis and recommendations on. In order to make this process more efficient, I decided to build a preprocessing pipeline, which would contain most of the preprocessing steps needed to prepare the data before creating a modeling pipeline. 

## Building A Preprocessing Pipeline

I began by creating some functions to replace some of the missing data in the dataset.

```
#create functions for preprocessing

# function to replace NaN's in the ordinal and interval data 
def replace_NAN_median(X_df):
    opinions = ['opinion_seas_vacc_effective', 'opinion_seas_risk', 'opinion_seas_sick_from_vacc', 'household_adults',
                'household_children']
    for column in opinions:
        X_df[column].replace(np.nan, X_df[column].median(), inplace = True)
    return X_df

# function to replace NaN's in the catagorical data     
def replace_NAN_mode(X_df):
    miss_cat_features = ['education', 'income_poverty', 'marital_status', 'rent_or_own', 'employment_status']
    for column in miss_cat_features:
        X_df[column].replace(np.nan, statistics.mode(X_df[column]), inplace = True)
    return X_df

# function to replace NaN's in the binary data                                
def replace_NAN_0(X_df):
    miss_binary = ['behavioral_antiviral_meds', 'behavioral_avoidance','behavioral_face_mask' ,
    'behavioral_wash_hands', 'behavioral_large_gatherings', 'behavioral_outside_home',
    'behavioral_touch_face', 'doctor_recc_seasonal', 'chronic_med_condition', 
    'child_under_6_months', 'health_worker','health_insurance']
    for column in miss_binary:
        X_df[column].replace(np.nan, 0, inplace = True)
    return X_df
```

 I then used a `[FunctionTransformer](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.FunctionTransformer.html#sklearn.preprocessing.FunctionTransformer)` on these functions so I could use them in my pipeline.
 
```
# Instantiate transformers

# I used functions instead of SimpleImputer as the functions preserved  the feature names 
# throughout the pipeline
NAN_median = FunctionTransformer(replace_NAN_median)
NAN_mode = FunctionTransformer(replace_NAN_mode)
NAN_0 = FunctionTransformer(replace_NAN_0)

```

I decided to use a `[ColumnTransformer](https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html#sklearn.compose.ColumnTransformer)` so that I could use pre-made packages to process my data. 
I scaled my numeric ordinal and interval data using `MinMaxScaler` (this standardized all of the responses between 0-1, which was handy as most of the data was binary.)
I used `[OneHotEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html#sklearn.preprocessing.OneHotEncoder)` in order to create dummy variables of the categorical features. 
I then used `remainder="passthrough"` so that any columns not specified in the `ColumnTransformer` would be left alone.

```
col_transformer = ColumnTransformer(transformers= [
    # I chose MinMaxScaler vs. StandardScaler in order to keep my data in the binary range (0-1)
    ("scaler", MinMaxScaler(), ['opinion_seas_vacc_effective', 'opinion_seas_risk',
                                'opinion_seas_sick_from_vacc', 
                                'household_adults', 'household_children']),
     
     # OHE catagorical string data
    ("ohe", OneHotEncoder(sparse = False, drop = "first"), ['age_group','education', 'race', 'sex', 
                                'income_poverty', 'marital_status', 'rent_or_own',
                                'employment_status', 'census_msa'])],
    verbose_feature_names_out = False,
    remainder="passthrough")
    
 ```
 
## Creating The Model Pipeline

All together, this was the preprocessing pipeline, using sklearns `[Pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html#sklearn.pipeline.Pipeline)`
 
 ```
 # Preprocessing Pipeline (Yey!)
preprocessing_pipe = Pipeline(steps=[
    ("NAN_median", NAN_median), 
    ("NAN_mode", NAN_mode),
    ("NAN_0", NAN_0),
    ("col_transformer", col_transformer)
    ])
    
 ```

I then used this pipeline as a sub-pipeline in our first model pipeline. For the first model I chose to use a `[LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression)` model. `LogisticRegression` classification models are great at handling binary dependent variables, which is what we have in our dataset, so it seemed like a good model to start with. 

```
logreg_optimized_pipe = Pipeline(steps=[("preprocessing_pipe", preprocessing_pipe),
                                    ("log_reg", LogisticRegression(solver = 'liblinear',
                                                                   random_state = RANDOM_STATE,
                                                                   C = 10, penalty= 'l2'))])
```

## Navigating a Multi-Layered Pipeline

After creating this pipeline, I want to be able to pull out the feature importances so we know which features performed best in the model. There are two parts of feature importances to pull out of the pipeline - the feature **names** and the feature importance **values**. Fortunately, `LogisticRegression` has a built-in attribute `coef_` which lets us pull the values of the features easily. To use this attribute, we first need to navigate the pipeline!


**insert code here**

One way of navigating the pipeline is using `.steps`. This lets us call the specific sections of the pipeline that we want. It’s important to remember that Python uses 0 based numbering, so the first item in our pipeline lists will always be indexed as 0, the second will be 1 and so on. For example, calling using `.steps[0]` will call the first part of our pipeline - the preprocessing pipeline. 

**insert code here**

### Getting Feature Importance Values

For our example, we can navigate into the *second* part of our pipeline using .`steps` and then call the `.coef_` attribute to get our coefficients.

*inset code*

Whoops! Seems like we didn’t quite get what we wanted - we need to further specify exactly where we want to go in the pipeline. By using `.steps[1][1]` we should first navigate to the model pipeline, and then within the model pipeline be directed specifically to the model we created. Lets try re-running the `coef_` attribute on this line of code. 

**insert code**

Perfect! 

### Getting Feature Importance Names

For the second part of feature importances we will need to find the feature names. The best way I found to do this was to navigate to the `OneHotEncoder` step in our preprocessing pipeline. Then we can call `.get_feature_names_out()` to pull out the feature names that will be put into the model in the next step of our pipeline. These names match up with the `coefficients` we found above. Here we will use a different method to navigate the pipeline `.named_steps`. This allows us to navigate the pipeline using the names we gave different sections of the pipeline. After using those names, we still can use Python’s 0 based numbering to pull out the specific sections we want to look at. 

**enter code here**

Great! Now let's put it all together.

## Plotting Feature Importances

**insert code here**

Now you know the basics of navigating multi-tiered pipelines.  




