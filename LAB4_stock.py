import pandas
from pandas import DataFrame
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

data =pandas.read_csv('F:\Machine Learning\Movie Prediction\Google_Stock_Price_Train - Google_Stock_Price_Train (1).csv')
#print(data.describe())
X = DataFrame(data, columns={'Close'})
y = DataFrame(data, columns={'Open'})
z = DataFrame(data, columns={'High'})
m = DataFrame(data, columns={'Low'})
k = DataFrame(data, columns={'Volume'})
 
plt.subplot(2,2,1)

regression = LinearRegression()
regression.fit(X,y)
print (regression.coef_)    #theta_1
print (regression.intercept_)  

plt.scatter(X, y, alpha=0.6)
plt.plot(X, regression.predict(X), color = 'yellow' , linewidth = 2)
plt.title('Open vs Close')
plt.xlabel('Close')
plt.ylabel('Open')
plt.xlim(0,2000)
plt.ylim(200, 1000)


plt.subplot(2,2,2)

regression = LinearRegression()
regression.fit(X,z)
print (regression.coef_)    #theta_1
print (regression.intercept_)  

plt.scatter(X, z, alpha=0.6)
plt.plot(X, regression.predict(X), color = 'red' , linewidth = 2)
plt.title('High vs Close')
plt.xlabel('Close')
plt.ylabel('High')
plt.xlim(0,2000)
plt.ylim(200, 1000)


plt.subplot(2,2,3)

regression = LinearRegression()
regression.fit(X,m)
print (regression.coef_)    #theta_1
print (regression.intercept_)  

plt.scatter(X, m, alpha=0.6)
plt.plot(X, regression.predict(X), color = 'red' , linewidth = 2)
plt.title('Low vs Close')
plt.xlabel('Close')
plt.ylabel('Low')
plt.xlim(0,2000)
plt.ylim(200, 1000)

plt.subplot(2,2,4)

regression = LinearRegression()
regression.fit(X,k)
print (regression.coef_)    #theta_1
print (regression.intercept_)  

plt.scatter(X, k, alpha=0.6)
plt.plot(X, regression.predict(X), color = 'red' , linewidth = 2)
plt.title('Volume vs Close')
plt.xlabel('Close')
plt.ylabel('Volume')
plt.xlim(0,2000)
plt.ylim(200, 7000000)

plt.show()

