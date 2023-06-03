import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OrdinalEncoder

class LinearRegression():

	def fit_dg(self, x, y, it, alpha, normalizacion=False):
		self.normalizacion = normalizacion
		self.alpha = alpha
		self.it = it
		self.x_a = x
		self.x = x
		self.y = y
		self.m = self.x.shape[0]
		if self.normalizacion:
			self.normalizacionDeCaracteristicas()
		self.x = np.concatenate((np.ones((self.m,1)),self.x), axis=1)
		self.n = self.x.shape[1]
		self.J_History = []
		self.T_History = []
		self.theta = np.zeros(self.n)
		for i in range(self.it):
			self.theta -= (self.alpha / self.m)*(np.dot(self.x, self.theta) - self.y).dot(self.x)
			self.J_History.append(self.funcionDeCoste(self.theta))
			self.T_History.append(self.theta.tolist())
		print('Theta elegido:',self.theta)
		print('Coste minimo dg: ',min(self.J_History))


	def fit_ec(self, x, y,normalizacion=False):
		self.normalizacion = normalizacion
		self.x_a = x
		self.x = x
		self.y = y
		self.m = self.x.shape[0]
		if self.normalizacion:
			self.normalizacionDeCaracteristicas()
		self.x = np.concatenate((np.ones((self.m,1)),self.x), axis=1)
		self.n = self.x.shape[1]
		self.theta = np.zeros(self.n)
		self.theta = np.dot(np.dot(np.linalg.inv(np.dot(self.x.T,self.x)),self.x.T),self.y)
		print('Theta elegido:',self.theta)
		print('Coste minimo ec: ',self.funcionDeCoste(self.theta))


	def funcionDeCoste(self, theta):
		h = np.dot(self.x,theta)
		J = (1/(2 * self.m)) * np.sum(np.square(h - self.y))
		return J


	def normalizacionDeCaracteristicas(self):
		self.mean = np.mean(self.x, axis=0)
		self.sigma = np.std(self.x, axis=0)
		self.x = (self.x-self.mean)/self.sigma


	def predict(self,x):
		print('X:',x)
		if self.normalizacion:
			x = (x-self.mean)/self.sigma
		x = np.insert(x,0,1)
		print("Resultado:",np.dot(x,self.theta))


	def graficaCosteInteraciones(self):
		plt.plot(np.arange(0,self.it),self.J_History)
		plt.xlabel('Iteraciones')
		plt.ylabel('Coste')
		plt.show()


	def grafica2D(self):
		x1 = np.vstack(self.x[:,1])
		plt.plot(x1,self.y,'x')
		d = np.vstack(np.arange(start=np.min(self.x),stop=np.max(self.x),step=0.1))
		y_hat = np.dot(np.concatenate([np.ones([d.shape[0],1]),d],axis=1),self.theta)
		plt.plot(d,y_hat,"-")
		plt.show()


	def grafica3D(self):
		x1 = np.vstack(self.x[:,1])
		x2 = np.vstack(self.x[:,2])
		fig = plt.figure()
		ax1 = fig.add_subplot(111,projection='3d')
		ax1.scatter(x1,x2,self.y,c='b')
		d = np.vstack(np.arange(start=np.min(self.x),stop=np.max(self.x),step=0.1))
		y_hat = np.dot(np.concatenate([np.ones([d.shape[0],1]),d,d],axis=1),self.theta)
		ax1.plot_wireframe(d,d,np.vstack(y_hat),color='black')
		ax1.scatter(-0.44604386,-0.22609337,np.dot([ 1.,-0.44604386,-0.22609337],self.theta),
			c='r',s=50)
		plt.show()


	def graficaCoste(self):
		# Primera grafica
		t1 = np.vstack(self.t_history[:,0])
		t2 = np.vstack(self.t_history[:,1])
		fig = plt.figure()
		ax1 = fig.add_subplot(121,projection='3d')
		d1 = np.linspace(np.min(t1)-5, np.max(t1)+5, 100)
		d2 = np.linspace(np.min(t2)-5, np.max(t2)+5, 100)
		d1, d2 = np.meshgrid(d1, d2)
		the = np.zeros((self.n,1), dtype='float64')
		j = []
		for m in range(100):
			jf = []
			for n in range(100):
				the[0,0], the[1,0]= d1[m,n],d2[m,n]
				jf.append(self.funcionDeCoste(the)[0])
			j.append(jf)
		j = np.array(j)
		ax1.plot_surface(d1,d2,j)
		ax1.scatter(t1, t2, self.J, c='r')
		ax1.set_xlabel("theta0")
		ax1.set_ylabel("theta1")
		ax1.set_zlabel("Coste")
		# Segunda grafica
		ax2 = fig.add_subplot(122)
		ax2.contourf(d1,d2,j)
		ax2.plot(t1,t2,c='r',marker = 'x')
		#plt.subplots_adjust(right=1)
		plt.show()


	def graficaMultiCaracteristicas(self):
		fig, axs = plt.subplots(round(self.n/3), 3, figsize=(6, 4))
		col = 0
		fil = 0
		for j in range(self.n-1):
			d = np.vstack(np.arange(start=np.min(self.x[:,j+1]),stop=np.max(self.x[:,j+1]),step=0.1))
			axs[fil,col].plot(self.x[:,j+1],self.y,'x')
			axs[fil,col].plot(d,self.theta[0]+d*self.theta[j+1],'-')
			col += 1 
			if col==3 or col==6:
				fil += 1
				col = 0	
		plt.show()


	def graficos(self):
		if self.n == 2:
			self.graficaCosteInteraciones()
			self.grafica2D()
			#self.graficaCoste()
		elif self.n == 3:
			self.graficaCosteInteraciones()
			self.grafica3D()
		else:
			self.graficaCosteInteraciones()
			self.graficaMultiCaracteristicas()


if __name__ == '__main__':
	data = pd.read_csv('Red2.csv', index_col=0)
	print(data.columns)
	#encoder = OrdinalEncoder()
	#encoder.fit(data[['Country', 'Region', 'Winery']])
	#data[['Country', 'Region', 'Winery']] = encoder.transform(data[['Country', 'Region', 'Winery']])
	x = data[['Country', 'Region', 'Winery', 'Rating', 'NumberOfRatings']].to_numpy()
	y = data['Price'].to_numpy()
	print(x.shape)
	print(y.shape)
	modelo = LinearRegression()
	modelo.fit_dg(x,y,2000,0.01,True)
	#modelo.fit_ec(x,y,True)
	modelo.graficos()
	x_p = [15.0,212.0,2425.0,4.3,174.0]#36.67
	modelo.predict(x_p)
	x_p = [20.0,201.0,2540.0,4.5,218.0]#61.81
	modelo.predict(x_p)
	x_p = [15.0,335.0,108.0,3.7,1171.0]#14.54
	modelo.predict(x_p)
