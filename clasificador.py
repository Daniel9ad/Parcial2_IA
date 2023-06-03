import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import optimize
from scipy.io import loadmat


class LogisticRegression():

	def sigmoid(self, z):
		z = np.array(z)
		return 1 / (1 + np.exp(-z))


	def fit_op(self, x, y, x_p, y_p, labels):
		#x = self.normalizacionDeCaracteristicas(x)
		#x_p = self.normalizacionDeCaracteristicas(x_p)
		x = np.concatenate((np.ones((x.shape[0],1)),x), axis=1)
		x_p = np.concatenate((np.ones((x_p.shape[0],1)),x_p), axis=1)
		self.x = x
		self.y = y
		self.x_p = x_p
		self.y_p = y_p
		self.m = self.x.shape[0]
		self.n = self.x.shape[1]
		self.labels = labels
		self.all_theta = []
		self.all_Cost = []
		for c in self.labels:
			initial_theta = np.zeros(self.n)
			res = optimize.minimize(self.costFunction,
        	                initial_theta,
        	                (self.x, (self.y==c)),
        	                jac=True,
        	                method='CG',
        	                options={'maxiter': 50})
			self.all_Cost.append(res.fun)
			self.all_theta.append(res.x)
		self.all_theta = np.array(self.all_theta)
		print('Thetas elegidos:',self.all_theta)
		print('Costes minimos op:',self.all_Cost)
		self.precisionCojuntoEntrenamiento()
		self.precisionCojuntoPrueva()
		self.muestra()


	def normalizacionDeCaracteristicas(self,x):
		mean = np.mean(x, axis=0)
		sigma = np.std(x, axis=0)
		#with np.errstate(divide='ignore', invalid='ignore'):
		x_norm = (x - mean) / sigma
		#x_norm = np.nan_to_num(x_norm, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
		return x_norm


	def separarConjunto(self, x, y):
		self.x = x[0:int(x.shape[0]*0.8),:]
		self.y = y[0:int(y.shape[0]*0.8)]
		self.x_p = x[int(x.shape[0]*0.8):,:]
		self.y_p = y[int(y.shape[0]*0.8):]


	def costFunction(self, theta, X, y):
		m = y.size
		grad = np.zeros(theta.shape)
		h = self.sigmoid(X.dot(theta.T))
		a = 1e-08
		J = (1 / m) * np.sum(-y.dot(np.log(h+a)) - (1 - y).dot(np.log(1 - h+a)))
		grad = (1 / m) * (h - y).dot(X)
		return J, grad


	def precisionCojuntoEntrenamiento(self):
		p = np.argmax(self.sigmoid(self.x.dot(self.all_theta.T)), axis = 1)+1
		print('Precision conjunto de entrenamiento:',self.x.shape,np.mean(p==self.y)*100,'%')


	def precisionCojuntoPrueva(self):
		p = np.argmax(self.sigmoid(self.x_p.dot(self.all_theta.T)), axis = 1)+1
		print('Precision conjunto de prueva:',self.x_p.shape,np.mean(p==self.y_p)*100,'%')


	def predict(self,x):
		print('X:',x)
		m = x.shape[0]
		x = np.concatenate([np.ones((m, 1)), x], axis=1)
		for i,c in enumerate(self.labels):
			print("Probabilidad de",c,':',self.sigmoid(np.dot(x,self.all_theta[i]))*100,'%')

		p = np.argmax(self.sigmoid(x.dot(self.all_theta.T)), axis = 1)+1
		print('Eleccion:',p)
		self.displayData(x[:,1:])


	def muestra(self):
		# Selecciona aleatoriamente 100 puntos de datos para mostrar
		rand_indices = np.random.choice(self.m, 9, replace=False)
		sel = self.x[rand_indices, :]
		sel = sel[:,1:]
		self.displayData(sel)


	def displayData(self, X, example_width=None, figsize=(10, 10)):
		"""
		Muestra datos 2D almacenados en X en una cuadrícula apropiada.
		"""
		# Calcula filas, columnas
		if X.ndim == 2:
		    m, n = X.shape
		elif X.ndim == 1:
		    n = X.size
		    m = 1
		    X = X[None]  # Promocionar a una matriz bidimensional
		else:
		    raise IndexError('La entrada X debe ser 1 o 2 dimensinal.')
		example_width = example_width or int(np.round(np.sqrt(n)))
		example_height = n / example_width
		# Calcula el numero de elementos a mostrar
		display_rows = int(np.floor(np.sqrt(m)))
		display_cols = int(np.ceil(m / display_rows))
		fig, ax_array = plt.subplots(display_rows, display_cols, figsize=figsize)
		fig.subplots_adjust(wspace=0.025, hspace=0.025)
		ax_array = [ax_array] if m == 1 else ax_array.ravel()
		for i, ax in enumerate(ax_array):
		    ax.imshow(X[i].reshape(example_width, example_width, order='F'),
		              cmap='Greys', extent=[0, 1, 0, 1])
		    ax.axis('off')
		plt.show()





if __name__ == '__main__':
	# Se carga el dataset que contiene imagenes de caracteres
	# donde cada pixel se convierte en una caracterista
	data = loadmat('emnist-letters.mat')
	entrenamiento = data['dataset'][0][0][0][0][0]
	prueva = data['dataset'][0][0][1][0][0]
	# Se extrae la x de prueva y entrenamiento, como tambien la y de prueva y entrenamiento
	x = entrenamiento[0][0:10000,:]
	y = entrenamiento[1].ravel()[0:10000]
	x_p = prueva[0][0:1000,:]
	y_p = prueva[1][0:1000]
	print(x.shape)
	print(y.shape)
	print(x_p.shape)
	print(y_p.shape)
	# Se crea una instancia de la clase LogisticRegression que contiene al modelo
	modelo = LogisticRegression()
	# Se llama al metodo que encuentra los parametros del modelo
	modelo.fit_op(x,y,x_p,y_p,[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26])
	# Se realiza una prueva 
	xPrueva = x_p[7:8,:]
	modelo.predict(xPrueva)#a