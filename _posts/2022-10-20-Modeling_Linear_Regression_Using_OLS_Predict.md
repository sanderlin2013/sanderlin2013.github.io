-----------------------------------------------------------------------------

# Modeling Linear Regression Using OLS .predict()

## Introduction

Hey! In my last post, I wrote about studying data science while having math anxiety. Today I’m going to write about a portion of my last project that I got stuck on, and try to explain where I got stuck and how I overcame it. Hopefully, if someone has the same issue, this post will be helpful.

## Dataset   

My latest project (github repository can be found [here](https://github.com/sanderlin2013/King-County-House-Sales); the jupyter notebook can be found [here](https://github.com/sanderlin2013/King-County-House-Sales/blob/main/Kings%20County%20House%20Sales%20Analysis%20.ipynb)) looked into the [King County House Sales dataset](https://www.kaggle.com/datasets/harlfoxem/housesalesprediction) in order to find information on how home renovations might increase the estimated value of homes for the magazine 'Home Owners Yearly'. The magazine wanted to put out an article on what renovations will or will not improve the value of middle class and upper middle class homes. After some initial data cleaning, the dataset included 21,255 homes with a mean sale price of $535,000.

### Questions
The analysis focused on three main questions:

-   **Will increasing the living area size lead to an associated increase in the value of the home?**
    
-   Will adding bedrooms or bathrooms lead to an associated increase in the value of the home?
    
-   Is the grade or condition rating of the house associated with the value of the home?
    

For this blog post, we're only going to focus on the *first* question. I was able to find a statistically significant association between a home's living area and its sales price, with an R^2 of .46. This means that the **living area explains 46% of the variability of home's sales price** - a hefty chunk! (See the model output below.)

![picture of sqft model](/images/base_model_log100.png)

With the living area of a home accounting for so much of the home's value, I thought it would be useful to use the model we created to predict how much a home's sales price would increase as the living area increased.

## Getting Stuck

Honestly, I got stuck on how to do this for a few weeks. I didn’t feel comfortable turning in the project until I could talk about the data not only descriptively, but also **make predictions** about what could happen if changes were actually made to a home!  While the code that I eventually wrote looks very simple, it took me a long time to figure out. I have to credit this [short informative tutorial](https://github.com/fbenamy/tutoring/blob/main/Create%20Simulated%20Data%20for%20Multiple%20Regression.ipynb) which explains multiple regression beautifully for really helping me along.

## Getting Un-Stuck

### Short Intro to Linear Regression and OLS

When building the initial model I used `statsmodels.regression.linear_model.OLS` in order to create a linear regression model using Ordinary Least Squares (OLS) (documentation [here](https://www.statsmodels.org/dev/generated/statsmodels.regression.linear_model.OLS.html)). Linear regression models show the relationship between a dependent variable (y) and at least one independent variable (x). When we do this using OLS, we draw this line so as to minimize the difference between the real or observed values and the predicted values.

### My Solution

In our case the dependent variable would be the home prices, and the independent variable is the home's living area size. In order to use OLS to predict home prices, we can use `statsmodels.regression.linear_model.OLS.predict` method (documentation [here](https://www.statsmodels.org/dev/generated/statsmodels.regression.linear_model.OLS.predict.html#statsmodels.regression.linear_model.OLS.predict)) by calling .predict on our model. But before we can use the method on the model we created, we have to create a matrix in such a way that the predict method can be called.

First we need to create a target vector for our independent variable, living area per 100 feet. To do this I used `numpy.arange`. Then we will use `statsmodel.add_constant` to turn our vector into a matrix, with the first column containing only 1’s (the constant that we added) and the second column going from 1 to 99.
```
#defining the upper and lower bound of price, and the spacing interval

target_variable_vector = np.arange(1, 100, 1)


# adding a constant to turn our vector into a matrix

target_variable_matrix = sm.add_constant(target_variable_vector)
```

If you have more than one independent variable, you’ll have to create other vectors and add them into the matrix ([example here](https://github.com/fbenamy/tutoring/blob/main/Create%20Simulated%20Data%20for%20Multiple%20Regression.ipynb)). This was actually the part I got stuck on! The statsmodels package requires a constant in the feature matrix when predicting. 
After creating our feature matrix, we then can put in our original model, use the `.predict` method and throw in our matrix.

```
# getting our predictions

log_results = log_model4.predict(target_variable_matrix)
```

Due to the assumptions of linear regression (more information about that [here](https://www.statology.org/linear-regression-assumptions/)), when preprocessing the data I took the natural log of the home sale prices. This lets us use the OLS model, but it also makes our data hard to interpret. Now that we’ve gotten what we want out of the model (the predicted values) let’s turn the prices back to normal using `numpy.exp()`, which will calculate the exponential value, and make our results much more understandable.
```
price_results = np.exp(log_results)
```

Finally, we get to the finale: turning this into a graph!

## Conclusion: Predicting Home Sales Prices from Living Area


```
# specify size of plot
fig, ax = plt.subplots(figsize=(10, 5))

# set plot limits and tick labels
ax.set_xlim(0, 4000000)
plt.xticks([0,1000000,2000000,3000000,4000000],['0', '1 Million', '2 Million',
               	'3 Million', '4 Million'])
#set up scatterplot
sns.scatterplot(x=df['price'], y=df['sqft_living'], hue = df['grade'], marker='d', palette = 'crest')

#change axis titles and heading
plt.title('Home Prices vs. Square Foot Living Space vs. Building Grade', fontsize=15)
plt.xlabel('Home Prices ($)', fontsize=13)
plt.ylabel('Square Footage', fontsize=13)

plt.tight_layout()
plt.show();
```

![final graph of predictions](/images/graph_homeprice_100sqft_living.png)



Hopefully this explanation has helped to explain how to run linear regression model’s using statsmodel OLS, and specifically how to use the `predict` method when there is only one independent variable.


