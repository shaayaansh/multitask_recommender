# multitask_recommender
Multitask training for a recommender system

This repository contains a MultiTaskNet model which is trained for predicting user interactions with a movie, and the score they assign to the movies. 
The model has two seperate heads, one for each task.
The first task is a regression task using an MLP with variable count of dense layers, and a ReLU layer after each dense layer. The inputs for this head is the a matrix $\Sigma$.



