import numpy as np

class FullyConnect:
		"""
		一次性输入多个数据的梯度，并将这些梯度求平均值，再使用这个平均
		值对参数进行更新。这样做可以利用并行计算来提高训练速度。我们一
		性一起计算的一组数据称为一个batch。同时，我们称所有训练图片都
		已参与一边训练的一个周期称为一个epoch。每个epoch结束时，我们
		会将训练数据重新打乱，通常会训练多个epoch
		"""
	def __init__(self, l_x, l_y):
		self.weights = np.random.randn(l_y, l_x) / np.sqrt(l_x)	#初始化权值
		self.bias = np.random.randn(l_y, 1)						#初始化偏移
		self.lr = 0												#初始化学习速率0

	def forward(self, x):
		self.x = x												#记录下中间结果以供反向传播时使用
		#计算全连接层的输出，x包含了多组数据，因此要分开计算
		self.y = np.array([np.dot(self.weights, x) + self.bias for xx in x])
		return self.y											#将计算结果向前传递

	def backward(self, d):	
		ddw = [np.dot(dd, xx.T) for dd, xx in zip(d, self.x)]	#得到对参数的梯度
		self.dw = np.sum(ddw, axis=0) / self.x.shape[0]			#对weights的导数
		self.db = np.sum(d, axis=0) / self.x.shape[0]			#对bias的导数
		self.dx = np.array([np.dot(self.weights.T, dd) for dd in d])									
		self.weights -= self.lr * self.dw
		self.bias -= self.lr * self.db
		return self.dx											#反向传播梯度

class Sigmoid:
	def __init__(self):
		pass
	
	def sigmoid(self, x):										
		return 1/(1+np.exp(-x))

	def forward(self, x):
		self.x = x
		self.y = self.sigmoid(x)
		return self.y

	def backward(self):											#返回x处梯度dx，也就是导数
		sig = self.sigmoid(self.x)
		self.dx = sig*(1-sig)
		return self.dx

def main():
	fc = FullyConnect(2, 1)										#创建网络
	sigmoid= Sigmoid()											#创建卷积项
	x = np.array([[1], [2]])									#得到测试输入
	print("weights:", fc.weights, 
		  'bias:', fc.weights, 
		  'input:', x)											#输出随机化的权值，偏移

	y1 = fc.forward(x)											#x向前传播得到y1
	y2 = sigmoid.forward(y1)									#y1运算得到y2
	print("forward result: ", y2)								#输出前向传播结果y2

	d1 = sigmoid.backward()												
	dx = fc.backward(d1)
	print("backward result: ", dx)
if __name__ == "__main__":
	main()