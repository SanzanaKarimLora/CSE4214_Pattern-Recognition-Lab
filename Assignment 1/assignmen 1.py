
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from google.colab import files
uploaded = files.upload()

import io
import pandas as pd
df = pd.read_csv(io.BytesIO(uploaded['train.txt'] ), delimiter =" ",names =['x','y', 'label'],  header = None)
print(df)

x = df['x'].tolist()
y= df['y'].tolist()
classes =df['label'].tolist()

class1Indexes =[]
class2Indexes =[]

for i in range(len(classes)):
     if classes[i]==1:
        class1Indexes.append(i)
     else:
        class2Indexes.append(i)

x_class1 =[]
x_class2 =[] 
y_class1 =[]
y_class2 =[]

for i in range(len(class1Indexes)):
    x_class1.append(x[class1Indexes[i]])
    

    
for i1 in range(len(class2Indexes)):
    x_class2.append(x[class2Indexes[i1]])

for i2 in range(len(class1Indexes)):
    y_class1.append(y[class1Indexes[i2]])
    

    
for i3 in range(len(class2Indexes)):
    y_class2.append(y[class2Indexes[i3]])

def plotMain():
     plt.scatter(x_class1,y_class1, color='r',marker='o',label='Train class 1')
     plt.scatter(x_class2,y_class2, color='b',marker='*',label='Train class 2')
     #plt.scatter(0,2, color='b')
     
        
        
plotMain()
plt.legend()
plt.show()

x_classnew =[]
y_classnew =[]


for i in range(len(x_class1)):
    x_classnew.append(x_class1[i])
    x_classnew.append(y_class1[i])
   

for i1 in range(len(x_class2)):
    y_classnew.append(x_class2[i1])
    y_classnew.append(y_class2[i1])

class1= (np.array(x_classnew)).reshape((len(x_classnew))//2,2)

class2= (np.array(y_classnew)).reshape((len(y_classnew))//2,2)

class1_Mean =np.mean(class1,axis=0)

class2_Mean = np.mean(class2, axis=0)

plotMain()

from google.colab import files
uploaded_test = files.upload()

import io
import pandas as pd
df_test = pd.read_csv(io.BytesIO(uploaded_test['test.txt'] ), delimiter =" ",names =['x','y', 'label'])
print(df_test)

u=[]
for i in range(len(df_test)):
     u.append(df_test['x'][i])
     u.append(df_test['y'][i])

testdatas= np.array(u).reshape(len(u)//2,2)

plotMain()

plt.plot(class1_Mean[0],class1_Mean[1],color='r',marker='D',label='Mean class 1')
plt.plot(class2_Mean[0],class2_Mean[1],color='b',marker='s',label='Mean class 2')  

m1 =[]


for i in range(len(testdatas)):
    x=testdatas[i,:]
    #print(x)
    #print(x[0])
    #print(x[1])
    g1 =abs(x.dot(class1_Mean.T)-0.5*(class1_Mean.dot(class1_Mean.T)))
    g2 =abs(x.dot(class2_Mean.T)-0.5*(class2_Mean.dot(class2_Mean.T)))
    
    if g1> g2:
        plt.plot(x[0],x[1], color='r',marker='+')
        m1.append(1)
    else:
        plt.plot(x[0],x[1], color='b',marker='v')
        m1.append(2)


     
        
plt.legend()
plt.show()







m1_actual= df_test['label'].tolist()

count1=0
for i in range(len(m1)):
    if m1[i]==m1_actual[i]:
        count1=count1+1

print("ACCURACY:", (count1/len(m1))*100 ,"%")

f=[]
a1 = df['x'].values
a2 = df['y'].values


for i in range(len(a1)):
    f.append(a1[i])
    f.append(a2[i])
    
a3 = df_test['x'].values
a4 = df_test['y'].values 

for i1 in range(len(a3)):
    f.append(a3[i1])
    f.append(a4[i1])

max_f=max(f)
min_f=min(f)

range1 = np.linspace(min_f,max_f,1000)

shaped_range= range1.reshape(len(range1)//2,2)

plotTest_Mean()
xvalues = np.arange(min_f, 7, 0.25)
constant = (np.dot(np.transpose(class1_Mean), class1_Mean) - np.dot(np.transpose(class2_Mean),class2_Mean))/2.0
coeff1 = np.dot(np.transpose(class1_Mean), np.array([1, 1]))
coeff2 = np.dot(np.transpose(class2_Mean), np.array([1, 1]))

yValues = []
for x in xvalues:
    yValues.append((coeff1 * x + constant) / coeff2)

plt.plot(xvalues, yValues, marker=".", color="g",label = "Decision_Boundary")
plt.legend()
plt.show()

