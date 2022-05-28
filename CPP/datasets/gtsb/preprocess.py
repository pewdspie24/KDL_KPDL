import os,glob,cv2 
import numpy as np
from skimage.transform import resize
images_train = glob.glob("/mnt/01D4B61792FFD5D0/btl_co_chi/src/datasets/gtsb/train/*/*.*")
x_train=[]
y_train=[]
np.random.shuffle(images_train)
import numpy as np
for image in images_train:
    im = cv2.imread(image)
    im = resize(im,(32,32))
    im=np.asarray (im).flatten() 
    x_train.append(im)
    y_train.append(int( image.split(os.path.sep)[-2]))

images_test=glob.glob("/mnt/01D4B61792FFD5D0/btl_co_chi/src/datasets/gtsb/validate/*/*.*")
x_test=[]
y_test=[]
for image in images_test:
    print(image)
    im = cv2.imread(image)
    im = resize(im,(32,32))
    im=np.asarray (im).flatten() 
    x_test.append(im)
    y_test.append(int( image.split(os.path.sep)[-2]))
# print(y_tra)
with open("./train.csv","w+") as f:
    for x,y in zip(x_train, y_train):
        x=[str(i) for  i in x]
        strx = ",".join(x) + "," + str(y) +"\n"
        f.write(strx)
with open("./test.csv","w+") as f:
    for x,y in zip(x_test, y_test):
        x=[str(i) for  i in x]
        strx = ",".join(x) + "," + str(y) +"\n"
        f.write(strx)