import pandas as pd
import numpy as np
from sklearn.linear_model import SGDRegressor
import matplotlib.pyplot as plt 

def modelo(dt):
    dt = dt.rename(columns = {'1991': 'uno', '1992': 'dos',  '1993':'tres', '1994':'cuatro'})
    x = dt.drop(columns = ['cuatro'])
    y = dt['cuatro']
    mod = SGDRegressor(loss = 'squared_loss',alpha = 0.001, max_iter = 10000)
    mod = mod.fit(x,y)
    print("Parámetros: ", mod.coef_)
    print("\n")
    
    return mod

#Entrenamiento 1:
print("** Prueba 1 **")
dt = pd.read_csv('Train.csv')
mod = modelo(dt)


#Prueba 1:
X = pd.read_csv('Test.csv')
X= X.rename(columns = {'1991': 'uno', '1992': 'dos',  '1993':'tres', '1994':'cuatro'})
y = X['cuatro']
X = X.drop(columns = ['cuatro'])

test = mod.predict(X)
print("R2:", mod.score(X,y))
print("Bias: ", abs(test - y).mean())
print("Varianza: ", np.var(test))
print("\n")


#Entrenamiento 2:
print("** Prueba 2 **")
dt = pd.read_csv('Train2.csv')
mod = modelo(dt)


#Prueba 2:
X = pd.read_csv('Test2.csv')
X= X.rename(columns = {'1991': 'uno', '1992': 'dos',  '1993':'tres', '1994':'cuatro'})
y = X['cuatro']
X = X.drop(columns = ['cuatro'])

test = mod.predict(X)
print("R2:", mod.score(X,y))
print("Bias: ", (test - y).mean())
print("Varianza: ", np.var(test))
print("\n")

#Predicciones:
print('Predicciones: ')
dt = np.array([0.02, 0.2, 0.1])
dt = pd.DataFrame([dt], columns = ['uno', 'dos', 'tres'])
print(mod.predict(dt))

dt = np.array([0.1, 0.05, 0.25])
dt = pd.DataFrame([dt], columns = ['uno', 'dos', 'tres'])
print(mod.predict(dt))


#Gráfica.
x_ax = range(len(test))
plt.plot(x_ax, y, linewidth=1, label="Y real")
plt.plot(x_ax, test, linewidth=1.1, label="Predicción")
plt.title("Y real vs Predicción")
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend(loc='best',fancybox=True, shadow=True)
plt.grid(True)
plt.show() 


