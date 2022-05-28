# Training iris

## Parameters:

- linear [4, 16, 32, 4]
- activation last: softmax 
- optimizer: ADAM 
- gradient_clip: [-100, 100]
- scheduler: Piece wise step : 100-200-250

## Results :
- best acc = 1. 

## Setup 

- change path file train.txt and test.txt by your path in file train_iris.cpp.
- compile file train_iris.cpp: g++ train_iris.cpp ./matrix/*.cpp - o train_irir
- run: ./train_irir

# Training GTSB

## Parameters:

- linear [32*32*3, 16, 32, 7]
- activation last: softmax 
- optimizer: ADAM 
- scheduler: Piece wise step : 20-50
- gradient_clip: [-100, 100]
## Results :
- best acc = 0.9507 

## Setup 
- first preprocess dataset from image to txt with preprocess.py
- change path file train.csv and test.csv by your path in file train_gtsb.cpp.
- compile file train_gtsb.cpp: g++ train_gtsb.cpp ./matrix/*.cpp - o train_gtsb
- run: ./train_gtsb 




