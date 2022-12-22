# Navigating Pipelines to Find Feature Importances

## Introduction
In my most recent project, I analyzed the CDC’s National 2009 H1N1 Flu Survey. The goal of this analysis was to create models that could find the most important features in the survey, so we could use those features to create a new survey on COVID-19 vaccine compliance. The original data included questions about both H1N1 and the seasonal flu vaccine compliance, but I choose to focus on only the seasonal flu. I figured that as COVID-19 vaccines and boosters become more routine like the seasonal flu vaccine, the season flu questions would be more relevant to the new survey. 

I wanted to be able to build and test a few different models, so I could choose the best one to base my analysis and recommendations on. In order to make this process more efficient, I decided to build a preprocessing pipeline, which would contain most of the preprocessing steps needed to prepare the data before modeling it. 

##Building Preprocessing Pipeline

I began by creating some functions to replace some of the missing data in the dataset.

**insert code here**

 I then used a `FunctionTransformer` on these functions so I could use them in my pipeline. 

**insert code here**

I decided to use a `ColumnTransformer` so that I could use pre-made packages to process my data. 
I scaled my numeric ordinal and interval data using `MinMaxScaler` (this standardized all of the responses between 0-1, which was handy as most of the data was binary.)
I used `OneHotEncoder` in order to create dummy variables of the categorical features. 
I then used `remainder="passthrough"` so that any columns not specified in the `ColumnTransformer` would be left alone.

**insert code here**

## Creating Model Pipeline

Here was the final pipeline:

**insert code here**

I then used this pipeline as a sub-pipeline in our first model pipeline. For the first model I chose to use a `LogisticRegression` model. `LogisticRegression` classification models are great at handling binary dependent variables, which is what we have in our dataset, so it seemed like a good place to start. 

**insert code here**
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




