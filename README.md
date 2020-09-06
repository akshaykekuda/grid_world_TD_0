# Value function approximation using TD(0) for Grid-World

## Grid World Problem

The cells of the grid correspond to the states of the environment. At each cell, four actions are possible: north, south, east, and west, which deterministically cause the agent to move one cell in the respective direction on the grid. Actions that would take the agent off the grid leave its location unchanged, but also result in a reward of −1. Other actions result in a reward of 0, except those that move the agent out of the special states A and B. From state A, all four actions yield a reward of +10 and take the agent to A’. From state B, all actions yield a reward of +5 and take the agent to B’.

![image](https://user-images.githubusercontent.com/22128902/92333059-fccd4a00-f09f-11ea-9498-019609bc0f1d.png)

The goal of the problem is to find the approximate value function w<sub>t</sub><sup>T</sup> φ(S<sub>t</sub>) and average reward t for various initial states. 
