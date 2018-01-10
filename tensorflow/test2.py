import tensorflow as tf


a = tf.add(3, 5)
#sess = tf.Session()
#print(sess.run(a))
#sess.close()
with tf.Session() as sess:
    print(sess.run(a))

x = 2
y = 3
op1 = tf.add(x, y)
op2 = tf.multiply(x, y)
op3 = tf.pow(op2, op1)
with tf.Session() as sess:
    op3 = sess.run(op3)
    print(op3)

with tf.device('/CPU:0'):
#安装好GPU之后，可以使用GPU
    a = tf.constant([[1.0]], name='a')
    b = tf.constant([[1.0]], name='b')
    c = tf.matmul(a, b)

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

print(sess.run(c))

