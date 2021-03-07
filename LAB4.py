import pandas
from pandas import DataFrame
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

data =pandas.read_csv('F:\Machine Learning\Movie Prediction\position_salary_1.csv')
#print(data.describe())
X = DataFrame(data, columns={'level'})
y = DataFrame(data, columns={'salary'})

regression = LinearRegression()
regression.fit(X,y)
print (regression.coef_)    #theta_1
print (regression.intercept_)   

plt.scatter(X, y, alpha=0.6)
plt.plot(X, regression.predict(X), color = 'yellow' , linewidth = 2)
plt.title('Level vs Salary')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()