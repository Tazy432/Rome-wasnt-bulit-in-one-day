
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
import numpy as np
import sympy as sp
# 0.5665446 -> 0.57 -> 57%
def floatTransformer(x):
    transformedFloat=f"{x:.2f}"
    transformedFloat=float(transformedFloat)*100
    return str(transformedFloat)+"%"

# 151. -->151
def floatPointEraser(yData):
    newData=[]
    for y in yData:
        result=(str(y).replace('.',""))
        result=result[0:len(result)-1]
        newData.append(result)
    return newData

#simple computation for the mean value
def mean(yData):
    records=0
    i=0
    for arg in yData:
        records=records+float(arg)
        i=i+1
    return float(records/i)

# computation of the sum of squares residuals around the mean ( for target y)
def ssMean(yData):
    meanAroundY=mean(yData)
    ssmean=float(0)
    for y in yData:
        ssmean=ssmean+(float((float(y)-meanAroundY)**2))
    return float(ssmean)

# computation of the sum of squares residuals around the fit ( for the regression line)
def ssFit(yData,xData,slope,intercept):
    #(data-line)^2
    ssfit=0
    for x,y in zip(xData,yData):
        ssfit=ssfit+float((float(y)-float(slope*x+intercept))**2)
    return float(ssfit)
# r^2 computation for finding how much of our variation in diabetes is a explained by our feature
def rSquared(ssfit,ssmean,stringX,stringY):
    r2=float(1-float(ssfit/ssmean))
    print(f"The variantion of {stringY} ,of a man , can be explined by {stringX} in proportion of {floatTransformer(r2)}")
    return r2

# simple liniar regression model using as feature the bmi for the Diabet data target
dateDiabet=load_diabetes(as_frame=True)
yDiabet=np.array(dateDiabet.target)
xBmi=dateDiabet.data['bmi']
a,b=sp.symbols(['a','b'])
mse=float(0)
for x,y in zip(xBmi,yDiabet):
    mse=mse+(a*x+b-y)**2
derivataA = sp.diff(mse, a)
derivataB = sp.diff(mse, b)
eq1 = sp.Eq(derivataA, 0)
eq2 = sp.Eq(derivataB, 0)
solution = sp.solve((eq1, eq2), (a, b))
print("The equation of the regression line is "+str(f"{solution[a]:.2f}")+"*x+"+str(f"{solution[b]:.2f}"))
xLinie=[-1,0,1]
yLinie=[-solution[a]+solution[b],solution[b],solution[a]*1+solution[b]]
plt.plot(xLinie,yLinie)
plt.scatter(xBmi,yDiabet)
plt.grid(True)
plt.show()

ssmean=ssMean(floatPointEraser(yDiabet))
ssfit=ssFit(floatPointEraser(yDiabet),xBmi,solution[a],solution[b])
rSquared(ssfit=ssfit,ssmean=ssmean,stringX="Body mass index",stringY="Diabetes ")

if __name__ == '__main__':
    pass

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
