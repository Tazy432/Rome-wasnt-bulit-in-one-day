
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
import numpy as np
import sympy as sp
dateDiabet=load_diabetes(as_frame=True)
yDiabet=np.array(dateDiabet.target)
xBmi=dateDiabet.data['bmi']
numeFeatures=np.array(dateDiabet.feature_names)
a,b=sp.symbols(['a','b'])
mse=float(0)
for x,y in zip(xBmi,yDiabet):
    mse=mse+(a*x+b-y)**2
derivataA = sp.diff(mse, a)
derivataB = sp.diff(mse, b)
eq1 = sp.Eq(derivataA, 0)
eq2 = sp.Eq(derivataB, 0)
solution = sp.solve((eq1, eq2), (a, b))
print(solution)
print("The equation of the regression line is "+str(solution[a])+"*x+"+str(solution[b]))
xLinie=[-1,0,1]
yLinie=[-solution[a]+solution[b],solution[b],solution[a]*1+solution[b]]
plt.plot(xLinie,yLinie)
plt.scatter(xBmi,yDiabet)
plt.show()

if __name__ == '__main__':
    pass

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
