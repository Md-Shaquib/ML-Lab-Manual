import pandas
from pandas import DataFrame
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

data =pandas.read_csv('F:\Machine Learning\Movie Prediction\Google_Stock_Price_Train - Google_Stock_Price_Train.csv')
#print(data.describe())
X = DataFrame(data, columns={'Volume'})
#y = DataFrame(data, columns={'Open'})
#z = DataFrame(data, columns={'High'})
m = DataFrame(data, columns={'Low'})
k = DataFrame(data, columns={'Close'})
 

plt.subplot(2,2,1)

regression = LinearRegression()
regression.fit(X,m)
print (regression.coef_)    #theta_1
print (regression.intercept_)

plt.scatter(X, m, alpha=0.6)
plt.plot(X, regression.predict(X), color = 'red' , linewidth = 2)
plt.title('Low vs Volume')
plt.xlabel('Volume')
plt.ylabel('Low')
plt.xlim(0,7000000)
plt.ylim(200, 1000)

plt.subplot(2,2,2)

regression = LinearRegression()
regression.fit(X,k)
print (regression.coef_)    #theta_1
print (regression.intercept_)  

plt.scatter(X, k, alpha=0.6)
plt.plot(X, regression.predict(X), color = 'red' , linewidth = 2)
plt.title('High vs Volume')
plt.xlabel('Volume')
plt.ylabel('Close')
plt.xlim(0,7000000)
plt.ylim(200, 1000)

plt.subplot(2,2,3)

regression = LinearRegression()
regression.fit(X,k)
print (regression.coef_)    #theta_1
print (regression.intercept_)  

plt.scatter(X, k, alpha=0.6)
plt.plot(X, regression.predict(X), color = 'red' , linewidth = 2)
plt.title('High vs Volume')
plt.xlabel('Volume')
plt.ylabel('Close')
plt.xlim(0,7000000)
plt.ylim(200, 1000)

plt.show()