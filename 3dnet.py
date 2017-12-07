import tensorflow as tf
x=2
y=3

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

a=tf.constant([2, 3],name='a')
b=tf.constant([[0,1],[2,3]],name='b')
y=tf.matmul(b,tf.reshape(a,[2,1]),name='mul')
x=tf.add(a,b,name='add')
with tf.Session() as sess:
	x,y=sess.run([x,y])
	print x,y