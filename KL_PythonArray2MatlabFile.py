#http://www.voidcn.com/article/p-shumwygl-qx.html
#https://blog.csdn.net/shanshangyouzhiyangM/article/details/85251683

from PIL import Image
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import scipy.io as io

img1=np.array(Image.open('0.jpg'))  #From image to array
img2=np.array(Image.open('35.jpg'))
img3=np.array(Image.open('65.jpg'))
img4=np.array(Image.open('100.jpg'))
img5=np.array(Image.open('182.jpg'))

mat_name='./20200528.mat'
mat=img5
io.savemat(mat_name,{'name':img5})



y1=img1.flatten()   #From array to Vector
y2=img2.flatten()
y3=img3.flatten()
y4=img4.flatten()
y5=img5.flatten()

KL_0 = scipy.stats.entropy(y1, y1)
print("KL of the same images:  ",KL_0)

t1=np.vstack((y1,y2))
t2=np.vstack((y3,y4))

mat_name_1='./20200528_y1.mat'
mat_name_2='./20200528_y2.mat'
mat_name_3='./20200528_y3.mat'
mat_name_4='./20200528_y4.mat'

mat=y1
io.savemat(mat_name_1,{'name',mat})
mat=y1
io.savemat(mat_name_2,{'name',y1})
mat=t1
io.savemat(mat_name_3,{'name',t1})
mat=t2
io.savemat(mat_name_4,{'name',t2})








