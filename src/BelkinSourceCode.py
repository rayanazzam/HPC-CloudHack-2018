
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 14:50:31 2018

@author: Belkin
"""

from __future__ import print_function
import matplotlib.pyplot as plt
import pandas
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import numpy as np


#read and shuffle data
df=pandas.read_csv("hack.csv",keep_default_na=False)
ds=df.sample(frac=1)

#rewrite shuffled data into new file
df.to_csv("test_data6.csv", sep=',', encoding='utf-8')
data=pandas.read_csv("test_data6.csv",keep_default_na=False)

#convert data into array, 2d array
array=data.values

# Split data to ones we want to use, and one we want to predict
x=array[1:,2:7]
y=array[1:,7]

#parse data into appropriate format
for i in range(len(y)):
    if(y[i]=='n/a' or y[i]==''):
        y[i]=5
    
    
for i in range(len(x)):
    for j in range(len(x[i])):
        if(x[i][j]=='n/a' or x[i][j]==''):
            x[i][j]=5

#split data into train and test sets

test_size=0.10
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=test_size)

model=XGBClassifier()
model.fit(x_train,y_train)

y_pred=model.predict(x_test)
predictions=[(float(value)) for value in y_pred]



avgPercentDiff=0
for i in range (len(predictions)):
    print(str(predictions[i]),"------------------------------->",str(y_test[i]))
    
    #do an average of the percent differences for prediction and actual value
    diff=(float(predictions[i])-float(y_test[i]))/float(y_test[i])
    avgPercentDiff+=diff
    
print("Average percent error of prediction is:{0:4.2f}%".format(avgPercentDiff/len(predictions)*100))


#prediction = pandas.DataFrame(predictions, columns=['predictions']).to_csv('prediction.csv')
#accuracy=precision_score(y_test,predictions)
#print("Accuracy:%2f%%" % (accuracy * 100.0))


#Changing each column into arraylist to make plotting easier
perfomance=array[1:,2]
cinema=array[1:,3]#the score for cinematography
script=array[1:,4]#the score for the script
plot=array[1:,5]#the score for the plot of the movie
mood=array[1:,6]#the score 
ImDB=array[1:,7]#the final IMDB rating for the movie



#defined a function to perform linear regression
#and calculate chi_squared values
def chi(x,y,title,xlab,ylab):
    for i in range(len(y)):
        if(y[i]=='n/a' or y[i]==''):
            y[i]=5
        else:
            y[i]=round(float(y[i]))
        
        
    for i in range(len(x)):
        if(x[i]=='n/a' or x[i]==''):
            x[i]=5
        else:
            x[i]=round(float(x[i]))
    
    sum_x=0
    sum_y=0
    sum_xx=0
    sum_xy=0
    chi_squared=0
   
    
    
    
    for value in x:
        sum_x=sum_x+value
    
    for value in y:
        sum_y=sum_y+value
        
    for value in x:
        sum_xx=sum_xx+value*value
    
    for i in range(len(x)):
        sum_xy=sum_xy+x[i]*y[i]
    
    #detrmine the cefficients A and B
    delta=len(x)*sum_xx-(sum_x)**2
    A=(sum_xx*sum_y-sum_x*sum_xy)/(delta)
    B=(len(x)*sum_xy-sum_x*sum_y)/delta
    
    #calculate chi_squared
    sigma_y=np.std(y)#standard deviation
    
    for i in range(len(y)):
        chi_squared=chi_squared+(y[i]-A-B*x[i])**2/sigma_y**2 
    
    sigma_A=sigma_y*np.sqrt(sum_xx/delta)
    sigma_B=sigma_y*np.sqrt(len(y)/delta)
    
    print("A={0:4.2f}+/-{1:4.2f},B={2:4.2f}+/-{3:4.2f}".format(A,sigma_A,B,sigma_B))
    
    
    #Calculate the Pearson r coefficient
    r=0
    sum_diffxy=0
    sum_xdiff_squared=0
    sum_ydiff_squared=0
    x_bar=np.mean(x)
    y_bar=np.mean(y)
    
    for i in range(len(y)):
        sum_xdiff_squared=sum_xdiff_squared+(x[i]-x_bar)**2
        sum_ydiff_squared=sum_ydiff_squared+(y[i]-y_bar)**2
        sum_diffxy=sum_diffxy+(x[i]-x_bar)*(y[i]-y_bar)
    r=sum_diffxy/np.sqrt(sum_xdiff_squared*sum_ydiff_squared)
    
    print("The correlation coefficient is r={0:6.2E} +/- {1:2.2f}".format(r,np.sqrt((1-r**2)/(len(ImDB)-2))))
    
    
    plt.figure(6)
    #plt.plot(x,y,yerr=np.std(y),fmt='bx',label='data',ecolor='red')
    xp=np.linspace(min(x),max(x),50)
    yp=A+B*xp
    plt.plot(xp,yp,'g-',label='fit')
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.title(title)
    plt.legend(loc='best')
    plt.show()

#function calls to test data
#chi(plot,ImDB,"ImDB rating vs plotScore","xPlot_Score","yImDB_rating")
#uncomment one by one to see plot for each catergory correlation with ImDB
chi(predictions,y_test,"Prediction Model","Prediction Value","Actual Value")
#chi(script,ImDB,"ImDB rating vs scriptScore","xScript_Score","yImDB_rating")
#chi(cinema,ImDB,"ImDB rating vs cinemaScore","xCinema","yImDB_rating")
#chi(perfomance,ImDB,"ImDB rating vs perfomanceScore","xPerfomance_Score","yImDB_rating")
#chi(mood,ImDB,"ImDB rating vs moodScore","xMood_Score","yImDB_rating")
