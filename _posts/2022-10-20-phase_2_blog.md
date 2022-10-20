# King County House Sales Analysis 

## Overview
Here we will be looking into the King County House Sales dataset to find information on how home renovations might increase the estimated value of homes (and by what amount) for the magazine 'Home Owners Yearly', who wants to put out an article on what renovations will or will not be likely to improve the value of middle class and upper middle class homes.


## Business Problem
The magazine gave us a few questions to focus on.
### Questions
- Will increasing the living area size lead to an associated increase in the value of the home?
- Will adding bedrooms or bathrooms lead to an associated increase in the value of the home?
- Is the grade or condition rating of the house associated with the value of the home?

## Data 

In order to do this, we will be looking at a data set on houses and housing prices from [King County in Washington State](https://en.wikipedia.org/wiki/King_County,_Washington).

### Dataset Size
After some initial data cleaning, we were left with 21,225 houses, whose mean sale price  was $535,000.

### Limitations of Dataset

There are some limitations inherent to this dataset. First and foremost, this dataset is all from King County, WA. This is a fairly affluent and densely populated area [(Wikipedia page)](https://en.wikipedia.org/wiki/King_County,_Washington), and as such the recommendations and conclusions from this data may not hold true for other areas with different characteristics (e.g. rural areas). More information and analysis is necessary to determine what neighborhoods and counties can use these recommendations. 

Additionally, there are many types of renovations that aren't included in the dataset (e.g. renovating the plumbing, new roof, adding a deck, ect.), which limits the specificity of the recommendations. 

### Why We Used This Dataset
Despite the above limitations, this dataset does represent a middle and upper class neighborhood, which is the demographic that the magazine is trying to appeal to. It does contain the information on bedrooms and bathrooms (which were some of the magazines specific questions that they wanted answers to) and was easily available. 

## Modeling

### Basic Model: Living Area Size
For our first basic model we used square footage of living space (`sqft_living`) as our independent variable and home prices as our dependent variable. The relationship between `sqft_living` and `price` did not meet the criteria for linear regression, so we ran a log function on both 'price` and `sqft_living` and discovered that using 'log_price` allowed us to run a linear regression model. 

### Bedrooms and Bathrooms
We initially tried to build off of the baseline model, but quickly discovered that due to high multicollinearity between `sqft_living` and `bathrooms` it was better to build a seperate model with `bedrooms` and `bathrooms` as the independent variables. Once again, `log_price` was our dependent variable.

### Grade and Condition
We OHE both `grade` and `condition` as they are both quantitative variables. As such, we first added all of the grade columns to our baseline model which included `sqft_living`. Afterwards, we added in the condition columns, creating our final model.

## Regression Results

### Living Area Size
#ADD IN BASELINE MODEL SUMMARY IMAGE
So we see here that there is a fairly large association between the log price of a home and its square footage of living space. This also has a small standard error, and confidence interval, making it a very accurate metric! As such, we can say that for every 100 square foot increase of living space in a home there is an association of an increase of .0396 of the log price. While this can be difficult to interpret in lay terms, it means all in all that based on what we see above there is a strong association between an increase in a home's square footage of living area and its price. Additionally, when we look at the R^2 we see that `sqft_living` can explain 46% of the `log_price` - a hefty chunk!

Graph of predicted home prices based compared to square foot living area: 
#ADD IN GRAPH IMAGE
### Bedrooms and Bathrooms
#ADD IN SECOND MODEL SUMMARY IMAGE
In examining this model, we see that adding bedrooms and bathrooms are both associated with an increase in the log price. The R^2 is lower than in our previous model (28.2%), which indicates that the number of bedrooms and bathrooms explains less of the log price than sqft_living. It's important to remember that there is likely collinearity between sqft_living and bedrooms and bathrooms, which could have led to the wonky results we saw in the analysis. That being said, we see that adding one bedroom is associated with a .05 (rounded) increase in log price, while adding one bathroom is associated with a .3 increase in log price - indicating that if you have to choose between adding a bedroom or a bathroom, adding a bathroom is indicated as the better fiscal choice. 

### Grade and Condition
Finally, let's look at our final model - the fifth_model, to look at grade and condition. Just a reminder, grade indicates the construction/building quality of the house, while condition refers to the maintenance level. 
#ADD FINAL MODEL IMAGE
At first glance, we see that the p-values of all of the conditions, except condition_5, indicate that these are not valuable contributors to the log price. From this, we can conclude that home maintenance only affects the sale price of a home if it is at the highest level. This makes sense, as it's usually assumed when one buys a home that some aspects will be run down and repairs will need to be made.

If one does maintain their home to this extent, ("All items well maintained, many having been overhauled and repaired as they have shown signs of wear, increasing the life expectancy and lowering the effective age with little deterioration or obsolescence evident with a high degree of utility") then there is an associated increase in log price of .2652.

Looking at the grade categories, we see that all of these categories are shown to be statistically significant. The coefficients of the grades increase as the grade increases, meaning that buildings with higher building grades are associated with higher log sale prices. 

## Conclusion 
Will increasing the living area size lead to an associated increase in the value of the home?
- Yes, with larger additions leading to larger increases in home prices. 

Will adding bedrooms or bathrooms lead to an associated increase in the value of the home?
- Yes, but bathrooms lead to a significantly larger increase.

Is the grade or condition rating of the house associated with the value of the home?
- Only the highest level of maintenance (condition) increases the sale price of a home
- Increasing the building quality is associated with higher home sale prices

### Possible Next Steps
- Look at data from other counties
- Look further into disentangling 
the collinearity between living space and 
bedrooms/bathrooms
- Investigate datasets with information 
on other renovations (plumbing, electric, ect.)




