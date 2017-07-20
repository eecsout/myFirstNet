#coding=utf-8
import numpy as np

class Data:
	"""
	数据层
	"""
	def __init__(self, name, batch_size):           #数据所在的文件名和每个batch中图片的数量batch_size
		with open(name, 'rb') as f:
			data = np.load(f)
		self.x = data[0]                            #输入x
		self.y = data[1]                            #对于每个x预期正确输出y
		self.l = len(self.x)                        #训练基地数目
		self.batch_size = batch_size                #batch的数目
		self.pos = 0                                #pos用来记录数据读取的位置，即第pos张图片

	def forward(self):                              #前向传播
		pos = self.pos								#记录pos
		bat = self.batch_size
		l = self.l
		if pos + bat >= l:                          #已经是最后一个batch了
			ret = (self.x[pos:l], self.y[pos:l])	#得到这些属性与label
			self.pos = 0							#将pos赋值0准备重新开始
			#下面四行代码用于打乱数据
			index = range(l)
			np.random.shuffle(index)
			self.x = self.x[index]
			self.y = self.y[index]
		else:                                       #不是最后一个batch, pos直接加上batch_size
			ret = (self.x[pos:pos + bat], 			#ret为训练对
			self.y[pos:pos + bat])
			self.pos += self.batch_size

		return ret, self.pos  						#返回的pos为0时代表一个epoch已经结束

	def backward(self, d):  						#数据层无backward操作
		pass


class FullyConnect:
	def __init__(self, l_x, l_y):  								#两个参数分别为输入层的长度和输出层的长度
		#随机数初始化参数
		self.weights = np.random.randn(l_y, l_x) / np.sqrt(l_x) #初始化一个l_y*l_x矩阵
		self.bias = np.random.randn(l_y, 1)
		self.lr = 0  											#先将学习速率初始化为0，最后统一设置学习速率

	def forward(self, x):
		"""
		这里传入的x是好多个测试数据的属性
		在书上读到的w0其实就是这里的bias
		y是每个传递来的数据得到的输出
		"""
		self.x = x  													#把中间结果保存下来，以备反向传播时使用
		self.y = np.array([np.dot(self.weights, xx) + self.bias for xx in x])
		return self.y  													#将这一层计算的结果向前传递

		"""
		如果是最后一级，接受的则是标准label和神经网络输出的差
		总之都是利用来自上一层的数据，然后对于点乘上这一层上次的计算结果
		"""
	def backward(self, d):												#d是这一级对x的导数
		ddw = [np.dot(dd, xx.T) for dd, xx in zip(d, self.x)]			#对x的导数乘以x得到对w的导数
		self.dw = np.sum(ddw, axis=0) / self.x.shape[0]					#求出ddw的平均值
		self.db = np.sum(d, axis=0) / self.x.shape[0]					#bias也要平均值
		self.dx = np.array([np.dot(self.weights.T, dd) for dd in d])	#下一级对x的导数

		# 更新参数
		self.weights -= self.lr * self.dw								#w = w + (-lr) * dw
		self.bias -= self.lr * self.db									#w0也一样，计算公式一样的
		return self.dx  												#反向传播梯度，传递给下一级


class Sigmoid:
	def __init__(self):  										#无参数，不需初始化
		pass

	def sigmoid(self, x):
		return 1 / (1 + np.exp(-x))

	def forward(self, x):
		self.x = x
		self.y = self.sigmoid(x)
		return self.y

	def backward(self, d):
		sig = self.sigmoid(self.x)
		self.dx = d * sig * (1 - sig)
		return self.dx  										#反向传递梯度，也就是对x的导数


class QuadraticLoss:
	"""
	这里最后一层是线性求和
	这里是损失函数层
	前向传输：
		我们有m个要分类的数据，有n个类，则x是m个列向量，是第i个数据是j个类则x(i,j)=1，否则等于0
		label是一个m*1的列向量，表示每个数据的类
		然后重构一个n*m的矩阵self.label，对于每个列向量有self.label[label[i]]=1,表示第i个数据的类为label[i]，
		self.label[i]最终是一个n*m矩阵，表示正确答案矩阵
		x与self.label:
		如果相同作差为0，否则为1，然后求平方和，再平均除以2
		返回损失值
	后向传输：
		self.dx等于输出值对和正确label的差向量的平均
	"""
	def __init__(self):
		pass

	def forward(self, x, label):								#输入传递到输出层的值和label，都是向量
		self.x = x												#x是一个列向量组
		self.label = np.zeros_like(x)  							#构造一个和x格式一样的零矩阵
		for a, b in zip(self.label, label):						#a是列向量，b是label值
			a[b] = 1.0
		self.loss = np.sum(np.square(x - self.label)) / self.x.shape[0] / 2
		return self.loss

	def backward(self):
		self.dx = (self.x - self.label) / self.x.shape[0]
		return self.dx


class CrossEntropyLoss:											#另一种损失函数
	def __init__(self):
		pass

	def forward(self, x, label):
		self.x = x
		self.label = np.zeros_like(x)
		for a, b in zip(self.label, label):
			a[b] = 1.0
		self.loss = np.nan_to_num(-self.label * np.log(x) - ((1 - self.label) * np.log(1 - x)))  # np.nan_to_num()避免log(0)得到负无穷的情况
		self.loss = np.sum(self.loss) / x.shape[0]
		return self.loss

	def backward(self):
		self.dx = (self.x - self.label) / self.x / (1 - self.x)  # 分母会与Sigmoid层中的对应部分抵消
		return self.dx


class Accuracy:													
	"""
	这个类用来计算神经网络的正确率
	传入x与label，label是这个输入所对应的正确结果
	x是经过神经网络分类后的结果
	对于每个x，提取出预测正确的实例
	"""
	def __init__(self):
		pass

	def forward(self, x, label):  													#只需forward
		self.accuracy = np.sum([np.argmax(xx) == ll for xx, ll in zip(x, label)])  	#对预测正确的实例数求和
		self.accuracy = 1.0 * self.accuracy / x.shape[0]							#正确实例/数据总数
		return self.accuracy														#返回正确率
