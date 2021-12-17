# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression

# US Population Growth

# Qa. import data
df_pop = pd.read_csv('us_pop_data.csv')

#Qa. create two new columns
df_pop['years_since_1790'] = df_pop['year'] - 1790
df_pop['population_in_millions'] = df_pop['us_pop']/1e06

# prep initial x and y values
x = df_pop['years_since_1790'].values[:,np.newaxis]
y = df_pop['population_in_millions'].values

# Qb. plot pop by millions versus years since 1790
plot1 = plt.figure(1)
plt.scatter(df_pop['years_since_1790'], df_pop['population_in_millions'])
plt.xlabel('Years since 1790')
plt.ylabel('Population in millions')


# Qc. create linear regression model
model = LinearRegression()
model.fit(x, y)
y_pred = model.predict(x)
r2_score(y, y_pred)
# Qc. Answer: R2-value = 0.9192437447080442

# Qd. create new column of years since 1790^2
df_pop['years_since_sq'] = df_pop['years_since_1790']**2

# Qe. linear regression of squared years since 1790
x2 = df_pop['years_since_sq'].values[:,np.newaxis]
model.fit(x2, y)
y2_pred = model.predict(x2)
r2_score(y, y2_pred)
# Qe. Answer: R2-value = 0.9984915694986646

# Qf. send models to plot
plt.plot(x, y_pred, c = 'r')
plt.plot(x, y2_pred, c = 'k')
# Qf. Answer: The years since 1790 squared model fits the data better. The R2-
# values indicate that the data fits the squared model better because -
# the squared R2 is 99.8%, whereas the "years since" R2 score is 91.9%.


# Customer Spending Data

# import data
df_spending = pd.read_csv('customer_spending.csv')

# Qa. make histogram
plot2 = plt.figure(2)
plt.hist(df_spending['ann_spending'])

# convert to np array
cs = np.array(df_spending['ann_spending'])

# Qb. make log transformed dataset
data = np.log(cs)

# Qc. make histogram of log transformed dataset
plot3 = plt.figure(3)
plt.hist(data)
plt.show()

# Qd. Answer: The log transformed data helps make the data less skewed. The - 
# first histogram is right skewed. The log transformed data can make patterns -
# more easily interpretable.  
