# Explaining ARIMA Time Series Models

![clock picture](/images/clock_banner.jpg)

## Introduction

Today we are going to do a quick high-level overview of ARIMA models which is one type of model used in time series analysis. ARIMA stands for Autoregressive Integrated Moving Average. The ARIMA model combines the strengths of Autoregressive (AR) and Moving Average (MA) models, while also accounting for trends in the data. Each of these components are mathematical equations where we specify at least one constant in order to model the data. Having all of these components allows for flexibility in the model.

## Autoregressive Model

Let's start off by talking about the AR (**Autoregressive**) part. The AR allows us to *predict something based on past observations*. This is vital in time series analysis as usually the whole objective of the time series is to be able to make predictions based on what has happened in the past! Whatever period we choose to use is called the AR order. In order to find this order, it is useful to look at the Partial Correlation Function (PACF). The PACF enables us to see what number of lags in the past would be relevant for our data today, so we can pick the best number for our AR order. Lags are essentially delays - if our lag is 3, that means we’re looking three time periods behind the current time period to predict it. 

Here’s an example - let's say we are a shop that sells catfish. Once a week we can put in an order of how many fish we want in the shop. In addition, people may want more fish on the weekends and holidays when they have more time to cook them. All in all, it may be best to use a 7 day lag (a week) in order to best predict how much we should order the next week, as predicting Friday’s sales based on Thursday may not be as predictive. 

You may ask “Why don’t we just use an AR order of 1, which would look at every datapoint we have and use all of them in prediction?” The issue with always using every data point is that it can sometimes lead to statistical issues, like overfitting our model to our data, which makes our predictions much worse. Additionally, using all of our data makes our model more complicated, which can be computationally expensive. In short, we want to find the time period in the past that would best predict our data today to make the simplest model that predicts our data well, and for different datasets that can be pretty different. 

## Moving Average Model

The next part of the ARIMA model we’ll talk about is MA (**Moving Average**). The MA helps to factor in random error found in real world data. It takes into account both the current error, as well as the past random error, depending on the order number that we specify. It calculates this error by finding the average of the data, and then for each specified period (e.g. if our order is 3, that could mean every three months in the past) finding how much the data differed from the average. As almost all real datasets include some random error, it’s important to factor this into our model! To find the right order for an MA model it can be useful to look at an Autocorrelation Function (ACF), which can help us find the number of lags we can use while representing the error and past errors in the data. 

## Problems with AR and MA Models: Stationarity 

The good news is that we can use AR and MA models together to create ARMA models. This allows us to use our data to predict what might happen in the future (AR) and take into account both the current and past compounded error in our models (MA). That sounds great! 

Unfortunately, both of these models have an a priori requirement in order to work properly; they need the data to be *stationary*. 

### What is stationary data? And why is it so important in time series modeling?

According to [Wikipedia](https://en.wikipedia.org/wiki/Stationary_process) “... a stationary process (or a strict/strictly stationary process or strong/strongly stationary process) is a stochastic process whose unconditional joint probability distribution does not change when shifted in time.[1] Consequently, parameters such as mean and variance also do not change over time. If you draw a line through the middle of a stationary process then it should be flat; it may have 'seasonal' cycles, but overall it does not trend up nor down. ”

While that’s a lot of statistical jargon, let’s focus on that last sentence: “If you draw a line through the middle of a stationary process then it should be flat; it may have 'seasonal' cycles, but overall **it does not trend up nor down**.”

Unfortunately, a lot of time series data can look something like [this](https://github.com/ritvikmath/Time-Series-Analysis/blob/master/SARIMA%20Model.ipynb): 


![time series with trend](/images/catfish_sales_with_trend.png)


What we see here is a clear upward trend in our data - and that trend really mucks up our model in a few ways. It makes it really hard to find the random error in the data (the MA component), and it can make our predictions worse, as just predicting the trend loses a lot of the nuance in the data (AR). In order for our data to be considered stationary we need a (relatively) constant mean and variance. 

## Stationarity in ARIMA: Integrated

While there are different ways to make data stationary, the **Integrated** (I) part of ARIMA model uses differencing to get rid of trends in our data. Assuming the data is following a linear trend, we know that the trend increases at a more or less constant rate. By specifying a specific time lag (the Integrated order number) we can subtract (or take the difference) from a specific time point the time point at the specified lag. 


Taking the difference can make our previous data with its trend look like [this](https://github.com/ritvikmath/Time-Series-Analysis/blob/master/SARIMA%20Model.ipynb): 


![time series without trend](/images/catfish_sales_no_trend.png)


Here we can see the data is no longer moving upwards, and seems to be centered around a mean - this allows us to use the AR and MA parts of our model.

## Conclusion

In this blog post we spoke about the three different parts that comprise an ARIMA model - the Autoregressive Integrated Moving Average model. AR allows us to use our past data to predict how our dataset might look in the future. The MA enables us to take into account the current and past random error in our dataset when making our model. Finally, the Integrated part of the ARIMA model takes the difference between a specified time point in the data and a previously defined time lag, which lets us remove trends in our data when needed to reach stationarity. 

For more information about time series modeling and ARIMA models I highly recommend [ritvikmath’s youtube series on the topic](https://www.youtube.com/playlist?list=PLvcbYUQ5t0UHOLnBzl46_Q6QKtFgfMGc3).
