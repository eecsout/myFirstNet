#coding=utf-8
from layers import *


def main():
	datalayer1 = Data('train.npy', 1024)            #用于训练，batch_size设置为1024
	datalayer2 = Data('validate.npy', 10000)        #用于验证，一次性计算所有的样例
	inner_layers = []
	inner_layers.append(FullyConnect(17 * 17, 20))	#增加隐层
	inner_layers.append(Sigmoid())					#隐层的Sigmoid运算
	inner_layers.append(FullyConnect(20, 26))       #增加一个隐层
	inner_layers.append(Sigmoid())
	losslayer = QuadraticLoss()						#增加损失层
	accuracy = Accuracy()

	for layer in inner_layers:
		layer.lr = 10000.0                          #为所有中间层设置学习速率（比较大）

	epochs = 20										#设置学习轮数
	for i in range(epochs):
		print('epochs:', i)							#输出这里是第几轮
		losssum = 0									
		iters = 0									#迭代次数为0
		while True:
			data, pos = datalayer1.forward()  		#从数据层取出数据
			x, label = data							#训练数据的属性和label
			for layer in inner_layers:  			#先是线性求和，然后计算Sigmoid
				x = layer.forward(x)				#得到要前向传输的数据

			loss = losslayer.forward(x, label)  	#调用损失层forward函数计算损失函数值
			losssum += loss							#计算出来损失函数
			iters += 1								#迭代次数+1
			d = losslayer.backward()  				#调用损失层backward函数曾计算将要反向传播的梯度

			for layer in inner_layers[::-1]:  		#反向传播，从后面开始，向前进行
				d = layer.backward(d)				#传递给后一个函数这层对x的导数

			if pos == 0:  							#一个epoch完成后进行准确率测试
				data, _ = datalayer2.forward()		#得到验证集的数据，一次全部验证，不需要pos
				x, label = data						#对于每个验证数据，提取输入值和label
				for layer in inner_layers:
					x = layer.forward(x)
				accu = accuracy.forward(x, label)  #调用准确率层forward()函数求出准确率
				print('loss:', losssum / iters)	   #损失函数的平均值
				print('accuracy:', accu)	 	   #输出正确率
				break							   #一个epoch完成，跳出while循环


if __name__ == '__main__':
	main()