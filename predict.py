import tensorflow as tf 
import numpy as np 
import cv2
import os

path = os.getcwd() +  '/model/'

img_test = 'test.jpg'

img_test = cv2.imread(img_test)

img_test = cv2.resize(img_test, (448,448))

img_test = np.reshape(img_test, (1,448,448,3))


sess = tf.InteractiveSession()

checkpoint = tf.train.latest_checkpoint(path)

graph = tf.get_default_graph()
saver = tf.train.import_meta_graph(checkpoint + '.meta')
saver.restore(sess, checkpoint)

output = graph.get_tensor_by_name('prediction:0')

out = sess.run(output, feed_dict={'X:0':img_test, 'dropout:0':1.0})

pred = np.asarray(out)

print(pred.shape)

print(pred[0][3:6])