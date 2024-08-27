# multitask_recommender
Multitask training for a recommender system. This is an answer to Standford CS330 Autumn 2023/2024 homework 0.

This repository contains a MultiTaskNet model which is trained for predicting user interactions with a movie, and the score they assign to the movies. </br>
The model has two seperate heads, one for each task:
  - The first task is a regression task using an MLP with variable count of dense layers, and a ReLU layer after each dense layer. The inputs for this head is the concatenation of three tensors: $[u_i| q_j| u_i * q_j]$ where $u_i * q_j$ is the elementwise product of user i vector represenation and the movie j vector representation.
  - The second task is a matrix factorization to predict the probability of a $user_i$ interacting with a $movie_j$ ==> $u_i^\mathbb{T}q_j + b_j$

the model has two functionalities to share the embedding layers between tasks and to not share the embeddings between tasks.



