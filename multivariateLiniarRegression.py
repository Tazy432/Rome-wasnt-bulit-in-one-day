import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
import numpy as np
import sympy as sp

dateDiabet = load_diabetes(as_frame=True)
yDiabet = np.array(dateDiabet.target)
xDiabet = np.array(dateDiabet.data)
a, b, c, d, e, f, g, h, k, l, o = sp.symbols('a b c d e f g h k l o')
mse = 0
slopes = [a, b, c, d, e, f, g, h, k, l]
for i in range(len(yDiabet)):
    y_pred = sum(slope * x for slope, x in zip(slopes, xDiabet[i])) + o
    squared_error = (y_pred - yDiabet[i]) ** 2
    mse += squared_error
derivatives = [sp.diff(mse, coef) for coef in [a, b, c, d, e, f, g, h, k, l, o]]
sistemEcuatii = [sp.Eq(deriv, 0) for deriv in derivatives]
solutie = sp.solve(sistemEcuatii, (a, b, c, d, e, f, g, h, k, l, o))
print(solutie)
