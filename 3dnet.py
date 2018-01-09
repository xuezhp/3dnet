import os
os.environ["CUDA DEVICES"]="3"
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import numpy as np
import visdom

import dataIO as d

class TDNet(object):
	def __init__(self,batch_size,num_epoch,objname):
		self.strides = [1,2,2,2,1]
		self.weights = {}
		self.cube_l = 64
		self.cube_w = 64
		self.cube_h = 64
		self.z_len = 200
		self.batch_size = batch_size
		self.num_epoch = num_epoch
		self.dis_thresholding = 0.8
		self.objname = objname
		self.obj_ratio = 0.7
		self.is_local = True
		self.beta1 = 0.5
		self.gen_lr = 0.0025
		self.dis_lr = 0.00001

		self.train_sample_directory='./sample/'
		self.model_directory='./models/'
		
	def build_model(self):
		xavier=tf.contrib.layers.xavier_initializer()
		self.weights['wg1']=tf.get_variable('wg1',shape=[4,4,4,512,self.z_len],initializer=xavier)
		self.weights['wg2']=tf.get_variable('wg2',shape=[4,4,4,256,512],initializer=xavier)
		self.weights['wg3']=tf.get_variable('wg3',shape=[4,4,4,128,256],initializer=xavier)
		self.weights['wg4']=tf.get_variable('wg4',shape=[4,4,4,64,128],initializer=xavier)
		self.weights['wg5']=tf.get_variable('wg5',shape=[4,4,4,1,64],initializer=xavier)

		self.weights['wd1']=tf.get_variable('wd1',shape=[4,4,4,1,64],initializer=xavier)
		self.weights['wd2']=tf.get_variable('wd2',shape=[4,4,4,64,128],initializer=xavier)
		self.weights['wd3']=tf.get_variable('wd3',shape=[4,4,4,128,256],initializer=xavier)
		self.weights['wd4']=tf.get_variable('wd4',shape=[4,4,4,256,512],initializer=xavier)
		self.weights['wd5']=tf.get_variable('wd5',shape=[4,4,4,512,1],initializer=xavier)

	def train(self):
		self.build_model()
		z_vec = tf.placeholder(shape=[self.batch_size,self.z_len],dtype=tf.float32)
		x_vec = tf.placeholder(shape=[self.batch_size,self.cube_l,self.cube_w,self.cube_h,1],dtype=tf.float32)

		# output of generator and discriminator
		gen_z_out = self.generator(z_vec)
		dis_z_out,dis_z_out_sigmoid = self.discriminator(gen_z_out)
		dis_z_out_sigmoid = tf.maximum(tf.minimum(dis_z_out_sigmoid,0.99),0.01)

		dis_x_out,dis_x_out_sigmoid = self.discriminator(x_vec,phase_train=True,reuse=True)
		dis_x_out_sigmoid = tf.maximum(tf.minimum(dis_x_out_sigmoid,0.99),0.01)

		# accurancy of discriminator
		acc_x = tf.divide(tf.reduce_sum(tf.cast(dis_x_out_sigmoid>0.5,tf.int32)),self.batch_size)
		acc_z = tf.divide(tf.reduce_sum(tf.cast(dis_z_out_sigmoid<0.5,tf.int32)),self.batch_size)
		dis_acc = 0.5*(acc_x+acc_z)


		# loss function
		gen_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(dis_z_out_sigmoid),logits=dis_z_out)
		dis_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(dis_x_out_sigmoid),logits=dis_x_out)\
		+tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(dis_z_out_sigmoid),logits=dis_z_out)
		gen_loss = tf.reduce_mean(gen_loss)
		dis_loss = tf.reduce_mean(dis_loss)

		gen_test_net = self.generator(z_vec,phase_train=False,reuse=True)

		para_g = [var for var in tf.trainable_variables() if any(x in var.name for x in ['wg', 'bg', 'generator'])]
		print para_g
		para_d = [var for var in tf.trainable_variables() if any(x in var.name for x in ['wd', 'bd', 'discriminator'])]
		print para_d
		optimizer_gen = tf.train.AdamOptimizer(learning_rate=self.gen_lr,beta1=self.beta1).minimize(gen_loss,var_list=para_g)
		optimizer_dis = tf.train.AdamOptimizer(learning_rate=self.dis_lr,beta1=self.beta1).minimize(dis_loss,var_list=para_d)

		vis = visdom.Visdom()
		saver = tf.train.Saver()
		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())

			volumes = d.getAll(obj=self.objname,train=True,is_local=self.is_local,obj_ratio=self.obj_ratio)
			print(self.objname)
			volumes = volumes[...,np.newaxis].astype(np.float)

			for epoch in range(self.num_epoch):
				ind = np.random.randint(len(volumes),size=self.batch_size)
				# training obj data
				x = volumes[ind]
				#noise vector
				z_sample = np.random.normal(0,0.33,size=[self.batch_size,self.z_len]).astype(np.float32)
				z = np.random.normal(0,0.33,size=[self.batch_size,self.z_len]).astype(np.float32)
				discriminator_loss = sess.run(dis_loss,feed_dict={z_vec:z,x_vec:x})
				generator_loss = sess.run(gen_loss,feed_dict={z_vec:z})
				dis_accuracy,x_acc,z_acc = sess.run([dis_acc,acc_x,acc_z],feed_dict={z_vec:z,x_vec:x})
				print 'dis_accuracy,x_acc,z_acc: ',dis_accuracy,x_acc,z_acc

				if dis_accuracy < self.dis_thresholding:
					sess.run([optimizer_dis],feed_dict={z_vec:z,x_vec:x})
					print('Training Discriminator1', 'epoch:', epoch, 'Dis_loss:', discriminator_loss,\
					 'Gen_loss:', generator_loss, 'Dis_acc:',dis_accuracy)

				sess.run([optimizer_gen],feed_dict={z_vec:z})
				print('Training Generator', 'epoch:', epoch, 'Dis_loss:', discriminator_loss,\
					'Gen_loss:', generator_loss, 'Dis_acc:',dis_accuracy)

				# generate objects
				if epoch % 10 ==0:
					print("sss----2")
					g_obj = sess.run(gen_test_net,feed_dict={z_vec:z})
					if not os.path.exists(self.train_sample_directory):
						os.makedirs(self.train_sample_directory)
					g_obj.dump(self.train_sample_directory+'/3dnet_'+str(epoch))
					id_ch = np.random.randint(0,self.batch_size,4)
					for i in range(4):
						if g_obj[id_ch[i]].max() > 0.5:
							print("save files")
							# d.poltVoxelVisdom(np.squeeze(g_obj[id_ch[i]]>0.5),vis,'_'.join(map(str,[epoch,i])))
							d.plotFromVoxels(np.squeeze(g_obj[id_ch[i]]>0.5),'Voxel_'+str(epoch)+'_'+str(i)+'.png')
							d.plotMeshFromVoxels(np.squeeze(g_obj[id_ch[i]]>0.5),threshold=0.5,filename='Mesh_'+str(epoch)+'_'+str(i)+'.png')
				if epoch % 10 == 10:
					print("sss----1")
					if not os.path.exists(self.model_directory):
						os.makedirs(self.model_directory)
					saver.save(sess,save_path=self.model_directory+'/3dnet_'+str(epoch)+'.cptk')

	def leaky_relu(self,x,alpha=0.2):
		return tf.maximum(x,alpha*x)

	def discriminator(self,tdobj,phase_train=True,reuse=False):
		with tf.variable_scope('discriminator',reuse=reuse):
			d1 = tf.nn.conv3d(tdobj,self.weights['wd1'],strides=self.strides,padding='SAME')
			d1 = tf.contrib.layers.batch_norm(d1,is_training=phase_train)
			d1 = self.leaky_relu(d1,alpha=0.2)

			d2 = tf.nn.conv3d(d1,self.weights['wd2'],strides=self.strides,padding='SAME')
			d2 = tf.contrib.layers.batch_norm(d2,is_training=phase_train)
			d2 = self.leaky_relu(d2,alpha=0.2)

			d3 = tf.nn.conv3d(d2,self.weights['wd3'],strides=self.strides,padding='SAME')
			d3 = tf.contrib.layers.batch_norm(d3,is_training=phase_train)
			d3 = self.leaky_relu(d3,alpha=0.2)

			d4 = tf.nn.conv3d(d3,self.weights['wd4'],strides=self.strides,padding='SAME')
			d4 = tf.contrib.layers.batch_norm(d4,is_training=phase_train)
			d4 = self.leaky_relu(d4,alpha=0.2)

			d5 = tf.nn.conv3d(d4,self.weights['wd5'],strides=[1,1,1,1,1],padding='VALID')
			d5_sigmoid = tf.sigmoid(d5)

		# print d5,d5_sigmoid
		return d5,d5_sigmoid

	def generator(self,z_vec,phase_train=True,reuse=False):
		with tf.variable_scope('generator',reuse=reuse):
			z_vec = tf.reshape(z_vec,(self.batch_size,1,1,1,self.z_len))
			# five convolutional layers
			g1 = tf.nn.conv3d_transpose(z_vec,self.weights['wg1'],(self.batch_size,4,4,4,512),strides=[1,1,1,1,1],padding='VALID')
			g1 = tf.contrib.layers.batch_norm(g1,is_training=phase_train)
			g1 = tf.nn.relu(g1)

			g2 = tf.nn.conv3d_transpose(g1,self.weights['wg2'],(self.batch_size,8,8,8,256),strides=self.strides,padding='SAME')
			g2 = tf.contrib.layers.batch_norm(g2,is_training=phase_train)
			g2 = tf.nn.relu(g2)

			g3 = tf.nn.conv3d_transpose(g2,self.weights['wg3'],(self.batch_size,16,16,16,128),strides=self.strides,padding='SAME')
			g3 = tf.contrib.layers.batch_norm(g3,is_training=phase_train)
			g3 = tf.nn.relu(g3)

			g4 = tf.nn.conv3d_transpose(g3,self.weights['wg4'],(self.batch_size,32,32,32,64),strides=self.strides,padding='SAME')
			g4 = tf.contrib.layers.batch_norm(g4,is_training=phase_train)
			g4 = tf.nn.relu(g4)

			g5 = tf.nn.conv3d_transpose(g4,self.weights['wg5'],(self.batch_size,64,64,64,1),strides=self.strides,padding='SAME')
			g5 = tf.nn.tanh(g5)

		# print 'g5', g5
		return g5

	def gen_3d(self,z):
		return self.generator(z,phase_train=False,reuse=True)

if __name__ == '__main__':
	test = False
	if test:
		pass
	else:
		tdnet = TDNet(32,1000,'chair')
		tdnet.train()


# op1=tf.add(x,y)
# print '1'
# op2=tf.multiply(x,y)
# print '2'
# op3=tf.pow(op1,op2)
# print '3'
# add_op=tf.add(x,y)
# print '4'
# mul_op=tf.multiply(x,y)
# print '5'
# useless=tf.multiply(x,add_op)
# print '6'
# pow_op=tf.pow(add_op,mul_op)
# print '7'
# with tf.Session() as sess:
# 	z,not_useless=sess.run([op3,useless])
# 	print z,not_useless

# a=tf.constant([2, 3],name='a')
# b=tf.constant([[0,1],[2,3]],name='b')
# y=tf.matmul(b,tf.reshape(a,[2,1]),name='mul')
# x=tf.add(a,b,name='add')
# with tf.Session() as sess:
# 	x,y=sess.run([x,y])
# 	print x,y