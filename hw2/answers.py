r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 (Backprop) answers

part1_q1 = r"""
### Question 1A
Let's consider an input $\mat{X}_{N \times f}$ where $N$ is the number of samples in the batch and $f$ is the number of
 features in each sample. Let's also consider a linear layer with the weight matrix $\mat{W}_{m \times f}$ where $m$ is the
 number of output features (e.g the number of units in the layer). Hence, the size of the layer's output matrix $\mat{Y} = \mat{X} \mat{W}^T$ is $N \times m$.  
We want to derive each element in $\mat{Y}$ by each element in $\mat{X}$.
 Therefore, the gradient $\pderiv{Y}{X}$ has the size  $N \times m \times N \times f$.  
Using the numbers given in the question, the size of the Jacobian is $64x512x64x1024$.

### Question 1B
This Jacobian is indeed sparse. This is because the linear model multiplies each input row by the weights, which
 produces the output row for that sample. In other words, $Y_{i,j} = X^{(i)} \cdot W^{(j)}$ where $X^{(i)}$ is the $i^{th}$ row of $\mat{X}$
 and $W^{(j)}$ is the $j^{th}$ row of $\mat{W}$. Note that $\pderiv{Y_{i,j}}{X}$ (the derivation of one element in $\mat{Y}$)
 is a matrix the same size as $\mat{X}_{64 \times 1024}$. Since $Y_{i,j}$ depends only on the elements in $X^{(i)}$, we get:
 $$
 \pderiv{Y_{i,j}}{\mat{X}}_{64 \times 1024}^{(k)} = \begin{cases}
\mat{W}^{(j)}, & k=i\\
0            ,  & \text{else}
\end{cases}
 $$
 E.g. The derivatives of each element in the output matrix that do not belong to
 the corresponding row in the input matrix are zero. Since the vast majority of these elements do not belong to the
 corresponding row, the vast majority of the elements in this tensor are zero, which makes the tensor sparse.  
 
### Question 1C
We **DO NOT** need to materialize the entire Jacobian since it is sparse. Instead of deriving $L$ by the entire matrix $\mat{X}$
 (e.g deriving by the elements of all samples), we can derive sample-by-sample.  
Note that:  
$$
\mat{Y}_{[1 \times 512]}^{(i)} = \mat{X}_{[1 \times 1024]}^{(i)} \cdot \mat{W}_{[1024 \times 512]}^T \\
\Rightarrow \pderiv{\mat{Y}^{(i)}}{\mat{X}^{(i)}} = \mat{W}
$$
Using the chain rule:
$$
\pderiv{L}{\mat{X}^{(i)}}_{[1 \times 1024]} = \pderiv{L}{\mat{Y}^{(i)}}_{[1 \times 512]} \cdot \pderiv{\mat{Y}^{(i)}}{\mat{X}^{(i)}}_{[512 \times 1024]}
$$ 
Since we are given $\pderiv{L}{\mat{Y}}$ and $\pderiv{L}{\mat{Y}^{(i)}} = \pderiv{L}{\mat{Y}}^{(i)}$,
$$
\delta \mat{X}^{(i)} = \pderiv{L}{\mat{X}^{(i)}} = \pderiv{L}{\mat{Y}}^{(i)} \cdot \mat{W}
$$
We can see that this way, instead of materializing a $64 \times 512 \times 64 \times 1024$ tensor, we only need the
 weight matrix $\mat{W}$ (which is already in memory) to perform gradient calculations of the loss for all samples.

### Question 2A
As before, we want to derive each element in $\mat{Y}_{N \times m}$ matrix by each each element in $\mat{W}_{m \times f}$.  
This is a $N \times m \times m \times f = 64 \times 512 \times 512 \times 1024$ tensor.
 
### Question 2B
This Jacobian is also sparse. As before,
$$
\pderiv{Y_{i,j}}{\mat{W}}_{512 \times 1024}^{(k)} = \begin{cases}
\mat{X}^{(i)}, & k=j\\
0            ,  & \text{else}
\end{cases}
$$
This is because the element $\mat{Y}_{i,j}$ is the dot product of the sample $X^{(i)}$ and the weights of
 the linear unit $W^{(j)}$, so the weights of any other linear unit are irrelevant for the calculation, and thus
 deriving by those elements results in 0.

### Question 2C
We **DO NOT** need to materialize the entire Jacobian. We can use the same trick as before - deriving the loss with
 respect to the weights of only one linear unit at a time. Here, we will look at the columns of $\mat{Y}$ instead of
 the rows (the "$(i)$" index will represent a column instead of a row). We get:
$$
\mat{Y}_{[64 \times 1]}^{(i)} = \mat{X}_{[64 \times 1024]} \cdot {\mat{W}^T}_{[1024 \times 1]}^{(i)} \\
\Rightarrow \pderiv{\mat{Y}^{(i)}}{\mat{W}^{(i)}} = \mat{X}
$$
Using the chain rule:
$$
\pderiv{L}{\mat{W}^{(i)}}_{[1 \times 1024]} = \pderiv{L}{\mat{Y}^{(i)}}_{[1 \times 64]} \cdot \pderiv{\mat{Y}^{(i)}}{\mat{W}^{(i)}}_{[64 \times 1024]} \\
\Rightarrow \delta \mat{W}^{(i)} = \pderiv{L}{\mat{W}^{(i)}} = \pderiv{L}{\mat{Y}}^{(i)} \cdot \mat{X}
$$ 
Where the index "$(i)$" is a row index in $\mat{W}$ (the weight of a single linear unit) 
 and a column index in $\pderiv{L}{\mat{Y}}$ (the derivatives of the loss concerning the same linear unit).


"""

part1_q2 = r"""
Back-propagation is **NOT REQUIRED** in order to train a neural network, since all of the partial derivatives can be
 calculated. However, this is a very inefficient process which consumes much more time and memory, so backpropagation
 is preffered.
"""


# ==============
# Part 2 (Optimization) answers


def part2_overfit_hp():
    wstd, lr, reg = 2e-5, 2e-2, 2e-1
    # Tweak the hyperparameters until you overfit the small dataset.
    return dict(wstd=wstd, lr=lr, reg=reg)


def part2_optim_hp():
    wstd, lr_vanilla, lr_momentum, lr_rmsprop, reg, = (
        1e-4,
        3e-4,
        3e-6,
        5e-5,
        1e-4,
    )

    # Tweak the hyperparameters to get the best results you can.
    # You may want to use different learning rates for each optimizer.
    return dict(
        wstd=wstd,
        lr_vanilla=lr_vanilla,
        lr_momentum=lr_momentum,
        lr_rmsprop=lr_rmsprop,
        reg=reg,
    )


def part2_dropout_hp():
    wstd, lr, = (
        1e-2,
        1e-5,
    )
    # Tweak the hyperparameters to get the model to overfit without
    # dropout.
    return dict(wstd=wstd, lr=lr)


part2_q1 = r"""
### Questions 1.1 and 1.2
We expect that increasing the dropout slightly will increase the generalization of the model, but too much dropout 
 will hamper the speed of the training process since less features are learnt in each epoch.  
The graphs match these expectations: 
* We can see that for `dropout=0` the accuracy on the training set is almost always larger than for the larger dropout
 values by a considerable margin, but on the test set the accuracy for `dropout=0.4` is usually higher. This implies
 overfitting for the `dropout=0` training process, hence the `dropout=0.4` generalizes better.
* The `dropout=0.8` graph is less accurate on both training and test sets for most epochs, but we can see that the
 accuracy on both sets gradually increases, and in the last epochs the accuracy on the test set is similar to the
 other two dropout settings. This matches out expectation for too much dropout.
"""

part2_q2 = r"""
Since the cross entropy loss is not upper-bound, a single very wrong prediction can potentially make
your loss increase. In that case, if we add this single wrong prediction, and two good predictions which 
"cost" less, the loss will increase while the accuracy will also increase. 

"""

part2_q3 = r"""
### Question 3.1
* Gradient Descent is a first-order iterative optimization algorithm for finding a local minimum of a differentiable
 function. The idea is to take repeated steps in the opposite direction of the gradient (or approximate gradient) of the
 function at the current point, because this is the direction of steepest descent.
* Back-Propagation is an algorithm for training artificial neural networks **using Gradient Descent**.
 the method calculates the gradient of the error function with respect to the neural network's weights.
 Taking advantage of the chain rule, calculation of the gradient proceeds backwards through the network,
 with the gradient of the final layer of weights being calculated first and the gradient of the first layer of weights
 being calculated last. This way, the gradients required for the Gradient Descent algorithm can be computed efficiently.
 
 
### Question 3.2
|                                     Gradient Descent (GD)                                    |                                                            Stochastic Gradient Descent (SGD)                                                           |
|:--------------------------------------------------------------------------------------------:|:------------------------------------------------------------------------------------------------------------------------------------------------------:|
|                 Error is summed over all samples before updating the weights                 |                              Samples are chosen at a random order. Weights are updated upon examining each training sample                             |
|             Expensive in terms of memory. Not suggested for huge training samples            |                                      Can be used for huge training samples since only one sample is run at a time                                      |
|                            Prone to fall into shallow local minima                           |                                                           Can avoid falling into local minima                                                          |
|                                No random shuffling is required                               |                                                       Random shuffling is required for each epoch                                                      |
| Given sufficient time to converge, result is optimal on the training set (can be an overfit) | Solution is good but not optimal on the training set since only one sample is examined when reaching the solution (can lead to better generalization)  |



### Question 3.3
* SGD can handle huge datasets, while GD requires loading the entire set to memory, which is not always possible.
* Depending on the initialization and learning rate, GD can converge to a shallow local minima (in deep leaning and
 machine learning in general, it is common that the loss function has many shallow minima). SGD can avoid falling into
 these, although it is not guaranteed.
* GD will converge to the optimal solution on the **training set**, which can lead to overfitting since the samples in
 the validation/test sets do not contribute to the loss. The fact that SGD uses a random sample each time can help
 prevent that.
"""


# ==============


# ==============
# Part 3 (MLP) answers


def part3_arch_hp():
    # Tweak the MLP architecture hyperparameters.
    n_layers = 4  # number of layers (not including output)
    hidden_dims = 60  # number of output dimensions for each hidden layer
    activation = "lrelu"  # activation function to apply after each hidden layer
    out_activation = "tanh"  # activation function to apply at the output layer
    return dict(
        n_layers=n_layers,
        hidden_dims=hidden_dims,
        activation=activation,
        out_activation=out_activation,
    )


def part3_optim_hp():
    #  - Tweak the Optimizer hyperparameters.
    #  - Choose the appropriate loss function for your architecture.
    #    What you returns needs to be a callable, so either an instance of one of the
    #    Loss classes in torch.nn or one of the loss functions from torch.nn.functional.

    import torch.nn
    import torch.nn.functional

    loss_fn = torch.nn.CrossEntropyLoss()  # One of the torch.nn losses
    lr, weight_decay, momentum = 0.01, 0.001, 0.9  # Arguments for SGD optimizer
    # lr, weight_decay, momentum = 7e-8, 1e-8, 0.0  # Arguments for SGD optimizer

    return dict(lr=lr, weight_decay=weight_decay, momentum=momentum, loss_fn=loss_fn)


part3_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""


part3_q4 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""
# ==============
# Part 4 (CNN) answers


def part4_optim_hp():
    import torch.nn
    import torch.nn.functional

    loss_fn = torch.nn.CrossEntropyLoss()  # One of the torch.nn losses
    lr, weight_decay, momentum = 0.01, 0.001, 0.9  # Arguments for SGD optimizer

    #  - Tweak the Optimizer hyperparameters.
    #  - Choose the appropriate loss function for your architecture.
    #    What you returns needs to be a callable, so either an instance of one of the
    #    Loss classes in torch.nn or one of the loss functions from torch.nn.functional.

    return dict(lr=lr, weight_decay=weight_decay, momentum=momentum, loss_fn=loss_fn)


part4_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

# ==============

# ==============
# Part 5 (CNN Experiments) answers


part5_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part5_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part5_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part5_q4 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part5_q5 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""
# ==============
