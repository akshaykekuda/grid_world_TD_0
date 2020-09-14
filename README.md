# TD(0) using value function approximation for Grid-World

## Grid World Problem

The cells of the grid correspond to the states of the environment. At each cell, four actions are possible: north, south, east, and west, which deterministically cause the agent to move one cell in the respective direction on the grid. Actions that would take the agent off the grid leave its location unchanged, but also result in a reward of −1. Other actions result in a reward of 0, except those that move the agent out of the special states A and B. From state A, all four actions yield a reward of +10 and take the agent to A’. From state B, all actions yield a reward of +5 and take the agent to B’.

![image](https://user-images.githubusercontent.com/22128902/92333059-fccd4a00-f09f-11ea-9498-019609bc0f1d.png)

The goal of the problem is to find the approximate value function w<sub>t</sub><sup>T</sup> φ(S<sub>t</sub>) and average reward t for various initial states.

## TD(0) using Linear Function approximation

All of the prediction methods involve updates to an estimated value function that shift its value at particular states toward a “backed-up value,” or update target, for that state. Let us refer to an individual update by the notation s->u, where s is the state updated and u is the update target that s’s estimated value is shifted toward. 
Machine learning methods that learn to mimic input–output examples in this way are called supervised learning methods, and when the outputs are numbers, like u, the process is often called function approximation. Function approximation methods expect to receive examples of the desired input–output behavior of the function they are trying to approximate. We use these methods for value prediction simply by passing to them the su of each update as a training example. We then interpret the approximate function they produce as an estimated value function.

We have the objective function, the Mean Squared Value Error denoted:

![image](https://user-images.githubusercontent.com/22128902/92333341-3c953100-f0a2-11ea-9eda-615d32400c8b.png)

Our goal is to minimize V̅E̅,  where **w** is the weight column vector. Stochastic Gradient Descent is a method of function approximation in value prediction, best suited for online updates.
In stochastic gradient-descent methods, the weight vector is a column vector with a fixed number of real valued components, **w** = (w<sub>1</sub>, w<sub>2</sub>,  . . . , w<sub>d</sub>)<sup>T</sup>, and the approximate value function v<sup>^</sup>(s, **w**) is a differentiable function of **w** for all s ε S.
Stochastic gradient-descent (SGD) methods do this by adjusting the weight vector after each example by a small amount in the direction that would most reduce the error on that example:

Stochastic gradient-descent (SGD) methods do this by adjusting the weight vector after each example by a small amount in the direction that would most reduce the error on that example:

![image](https://user-images.githubusercontent.com/22128902/92333561-1375a000-f0a4-11ea-8730-4cb509774584.png)

Suppose Ut is a random approximation of V<sub>π</sub>(S<sub>t</sub>), we can approximate it by substituting Ut in place of V<sub>π</sub>(S<sub>t</sub>). This gives the below equation:

![image](https://user-images.githubusercontent.com/22128902/92333586-4d46a680-f0a4-11ea-8d9e-1ebf3d9a5f9e.png)

**A semi gradient TD(0)** , uses the update target U<sub>t</sub> as:

![image](https://user-images.githubusercontent.com/22128902/92333593-6cddcf00-f0a4-11ea-8dfb-e174e2b0acbf.png)

When we use **Linear Function approximation**, TD(0) converges to the local minimum of V̅E̅.  
In the case of Linear function approximation, v(s, **w**) is a linear function of the w vector. Here, **x** is called the feature vector of state s: x(s) = (x<sub>1</sub>(s), x<sub>2</sub>(s), ….., x<sub>d</sub>(s))<sup>1</sup>T.  with same dimensionality as **w**. 

![image](https://user-images.githubusercontent.com/22128902/92333686-17ee8880-f0a5-11ea-95d8-6dc483dc3ec8.png)

Thus the weight update for SGD becomes:

![image](https://user-images.githubusercontent.com/22128902/92333696-2dfc4900-f0a5-11ea-992c-85ef3b11f89c.png)


## TD(0) update algorithm
R̅<sub>t+1</sub> = R̅<sub>t</sub> + β<sub>t</sub> (R̅<sub>t+1</sub> - R̅<sub>t</sub> )

w<sub>t+1</sub> = w<sub>t</sub> + α<sub>t</sub> δ<sub>t</sub> φ(S<sub>t</sub>)

where TD error, δ<sub>t</sub> = R<sub>t+1</sub> - R̅<sub>t+1</sub> + w<sub>t</sub><sup>T</sup> φ(S<sub>t+1</sub>) - w<sub>t</sub><sup>T</sup>φ(S<sub>t</sub>)


 
