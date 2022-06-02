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
 
 
 
### Question 3.4.A
Lets recall the equation for the gradient descent algorithm:
$$
\vec{\theta} \leftarrow \vec{\theta} - \eta \nabla_{\vec{\theta}} L(\vec{\theta}; \mathcal{D})
$$
where $\mathcal{D} = \left\{ (\vec{x}^i, \vec{y}^i) \right\}_{i=1}^{M}$ is our training dataset or part of it.  
  
Note that if $1 < M < N$, a backward pass on each $i$ will make the algorithm mini-batch gradient descent. The suggested
 approach does not backward-pass each mini-batch, so it is not the mini-batch GD algorithm (where the gradient is not
 equivalent to GD). In the suggested approach, the gradient will equal the gradient in GD if:
$$
L\left(\theta, \mathcal{D} = \left\{ (\vec{x}^i, \vec{y}^i) \right\}_{i=1}^{N}\right) = \sum_{j=1}^{m=N/k} L\left(\theta, \mathcal{D} = \left\{ (\vec{x}^i, \vec{y}^i) \right\}_{i=j \cdot k}^{(j+1) \cdot k}\right)
$$
E.g the loss for a forward pass on the entire dataset is equal to the sum of the losses for all minibatches in the dataset.  
If the above condition is true, then:
$$
\nabla_{\theta}L(\theta, D) = \sum_j \nabla_{\theta}L(\theta, D_j)
$$
E.g the gradient with respect to $\theta$ on the entire set is equal to the sum of the gradients on the mini-batches).
 In many loss functions this is usually the case, but since the loss function was not defined in the question we cannot 
 determine the suggested method will produce the same gradient with full certainty.
 
### Question 3.4.B
For each forward pass, we need to store some data to be used in the backward pass. The memory was depleated probably due
 to saving this data for all previous forward passes.

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
* Our model **Does Not** have a high optimization error. Based on the loss and accuracy graphs for the training set,
 it seems that the model has converged to a local minima, and more epochs will not improve the results by much. 
 
* Our model **has** a considerable generalization error. We can see from
 the loss graph that there is a considerable gap between the average loss on the training set and the average loss on
 the test set. We can also see that from the accuracy graph. This clearly suggests that the model was overfitted to the
 training set, so a generalization error does exists.
 
* Our model **Does** have a high approximation error. It can be seen clearly from the accuracy graph on the test set,
 which does not stabilizes even in the last epochs.
"""

part3_q2 = r"""
Looking at the graphs of the generated datasets, we can clearly see that there are more "class 1" (orange) samples in
 areas which are dominated by "class 0" (blue) samples, but not the other way around. Since the training process will
 try to minimize the loss on the generated samples, we expect that the model will classify samples in a blue dominated
 area as "0" and in orange dominated areas as "1". Since there are many more "1"s in "0" dominated areas than the other
 way around, **we expect a higher rate of false negatives than of false positives**. This is true for all sets,
 including the validation set.
"""

part3_q3 = r"""
In neither case we would choose the "optimal" point on the ROC curve.
1. In the case where the illness causes non-lethal symptoms, the cost and the risk of the second test do not justify
 sending borderline cases to do it. We would choose a point on the ROC curve which increases the False negative rate,
 (which may result in decreasing the true positive rate). In this way, only clear cases will be sent to the second test.
 
2. In the case where the illness will cause death with high probability, it is better to send borderline cases to the
 second test in order to increase the probability of saving lives. We would choose a point on the ROC curve that
 increases the false positive rate (and may decrease the true negative rate), in order to try to decrease the number
 of cases in which the illness is not diagnosed in a patient.
"""


part3_q4 = r"""
### Questions 4.1 + 4.2
We can clearly see that the model's accuracy increases with the width of each layer. The decision boundary is
 considerably more accurate when moving from a lowe width to a higher width. This is true for all depths. 
This does not imply that increasing the depth does not matter at all, as we can see improvement when increasing the
 number of layers, but for each depth, a big "jump" in accuracy occurs when increasing the width of each layer.

### Question 4.3
* Both models have similar accuracies (the deeper model has a slightly better accuracy on both validation and 
 test sets), but we can clearly see the difference in the decision boundary plot, which is more accurate for the deeper
 model. This implies that the deeper model generalizes better.
 
* A similar phenomenon occurs in the second case.

The more accurate decision boundary on the deeper model in both cases (although the number of parameters is the same)
 can be explained by the fact that deeper model can learn more complex features and thus the possibility for a
 more accurate decision boundary increases with the depth of the network.

### Question 4.4
By plotting the decision boundary on the training set, we can see that without changing the threshold, the decision
 boundary will (over) fit the training samples, but will not be as accurate on the validation set. By using the
 validation set for threshold selection, we increase the model's performance on data that the model was not
 trained with, leading to better generalization. For this reason, selecting the optimal threshold also improves the
 results on the test set.
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
### Question 1.1
Bottleneck block:
$$
\text{number of parameters} = \underbrace{\overbrace{1 \times 1}^{\text{kernel size}} \times \overbrace{256}^{\text{channels in}}
\times \overbrace{64}^{\text{channels out}} + \overbrace{64}^{\text{bias}}}_{1^{st} \text{conv layer}} + 
\underbrace{\overbrace{3 \times 3}^{\text{kernel size}} \times \overbrace{64}^{\text{channels in}}
\times \overbrace{64}^{\text{channels out}} + \overbrace{64}^{\text{bias}}}_{2^{nd} \text{conv layer}} + 
\underbrace{\overbrace{1 \times 1}^{\text{kernel size}} \times \overbrace{64}^{\text{channels in}}
\times \overbrace{256}^{\text{channels out}} + \overbrace{256}^{\text{bias}}}_{3^{rd} \text{conv layer}} = 70,016
$$
Regular block:
$$
\text{number of parameters} = \underbrace{\overbrace{3 \times 3}^{\text{kernel size}} \times \overbrace{256}^{\text{channels in}}
\times \overbrace{64}^{\text{channels out}} + \overbrace{64}^{\text{bias}}}_{1^{st} \text{conv layer}} + 
\underbrace{\overbrace{3 \times 3}^{\text{kernel size}} \times \overbrace{64}^{\text{channels in}}
\times \overbrace{256}^{\text{channels out}} + \overbrace{256}^{\text{bias}}}_{2^{nd} \text{conv layer}} = 295,241
$$


### Question 1.2
A single convolution kernel with the size of $ k \times k \times c $ will perform $k \times k \times c$ multiplications
 and $k \times k \times c - 1$ additions (with the bias parameter we get the same number of multiplications and
 additions). Note that floating point multiplication can be faster than floating point addition, depending on the
 architecture of the processor, so for the sake of this question we will treat them as having the same time.
So, for each input pixel (across all channels), we perform $ 2 \times k \times k \times c $ FP operations.
We'll mark $S_{[\text{num of pixels}]}$ as the number of the pixels in the input image, E.G the number of times we need to
 apply the convolution filter.  
Bottleneck block:
$$
\text{num of FP operations} = 2 \times \overbrace{1^2}^{\text{filter size}} \times \overbrace{256}^{\text{input channels}} \times \overbrace{64}^{\text{output channels}} 
\times S + 2 \times \overbrace{3^2}^{\text{filter size}} \times \overbrace{64}^{\text{input channels}} \times \overbrace{64}^{\text{output channels}} \times S +
2 \times \overbrace{1^2}^{\text{filter size}} \times \overbrace{64}^{\text{input channels}} \times \overbrace{256}^{\text{output channels}} \times S \\
= 2 \times S \times 69,632 = 139,264 \times S
$$
Regular Block:
$$
\text{num of FP operations} = 2 \times \overbrace{3^2}^{\text{filter size}} \times \overbrace{256}^{\text{input channels}} \times \overbrace{64}^{\text{output channels}} \times S +
2 \times \overbrace{3^2}^{\text{filter size}} \times \overbrace{64}^{\text{input channels}} \times \overbrace{256}^{\text{output channels}} \times S = 
4 \times 9 \times 64 \times 256 \times S = 589,824 \times S
$$

### Question 1.3
The bottleneck has 1 3x3 layer and 1 1x1 convolutions which does not provide spatial abilities across feature maps
unlike the standard block which has 2 3x3 convolution layers. 
since bottleneck reduces the number of channels and combines across multiple channels  it combines better across feature maps
unlike the standard block which doesnt alter the number of channels.
"""
#  2 \times S \times 69632 = 139,264 \times S

# ==============

# ==============
# Part 5 (CNN Experiments) answers


part5_q1 = r"""
### Question 1.1
We can see that for `K=32` and for `K=64`, increasing the depth did not necessarily increase the accuracy on the test
 set. We can see that for both Ks `L=8` achieved the best accuracy overall, with L16 close behind.  
 Notice that when increasing L, the gap between the train accuracy to the test accuracy decreases. This means that
 the deeper networks are less overfitted, on in other words, generalize better.
 Another seen to notice - it can clearly be seen that the learning rate was slower for each increased depth. 
 The implications of this can be seen in all of the graphs, where L4 and L8 took less time to converge 
 (E.G reach early stopping). The odd case being L2, since it learned slower than L4 and L8 but faster than L16.
 
### Question 1.2
There were no L values for which the network was not trainable. It did happen for L=16 when the learning rate was
 0.01, but when we lowered it to 0.001 it became trainable. 
"""

part5_q2 = r"""
In this experiment, for the sake of saving time, we lowered the batch size to 100 (350 before) and the early stopping
 to 3 (5 before). The results were sometimes still comparable to the results from the previous section, (which had
 a larger batch size and more epochs to train), but mostly the accuracies were lower than the previous experiment for
 the same configuration. In the previous configuration, early stopping could sometimes take ~40 minutes for each
 configuration with relatively small number of parameters (much more for larger configurations), 
 and we could not afford it. This shows the importance of training with larger batch sizes and enough
 epochs to escape local minima.  
   
In this experiment we can see that brute force works. Increasing the amount of layers while increasing the number of
 filters in each layer generally results in better accuract on the test set. The exception is with L=2 and K>=64,
 which for some reason gets much lower accuracy than the rest of the configurations. The best accuracy is achieved in
 the configuration of L=8 K=256. However, this configuration is very costly to train, and the difference between its
 accuracy and the accuracies of some of the other configurations which have much less parameters is 
"""

part5_q3 = r"""
In this experiment we had to change a few parameters in order for all the configurations to train properly, which may have caused some negative performance impact. The changes were:
* We had to lower the learning rate in order for the configurations with the most parameters to converge.
* We had to manipulate the `pool_every` parameter in order for the L4 network to be valid.
* We lowered the batch size to train faster.

As we can see in the graph, most of the configurations achieved approximately the same accuracy on the test set, but in a similar way to the previous experiment, when we increased L it took more epochs for the configuration to reach early stopping. Also, the larger the L, the lower the gap between training accuracy and test accuracy, which implies that the deeper models are less prone to overfitting.
"""

part5_q4 = r"""
We can clearly see that with the current configuration of ResNet and the training parameters we used, we got worse accuracies than most of the previous experiments. Like in experiment 1.3, a low value of `pool_every` caused the deeper networks to be invalid, and so this value was raised for the deeper models. 

It can clearly be seen from the graphs that all of the ResNet configurations in this experiment are prone to overfitting, since the gap between the test accuracy is very high, and can even reach almost to 50% in some of the cases.  

We can dee that the deeper the model, 
"""

part5_q5 = r"""
**Your answer:**
1.The first architecture we attempted to implement was the non-naive inception block module architecture. 
where each block contains 4 paths. 
the first path contains a 1x1 convolution  and a normalization afterwards
the second path contains two instances of 3x3 convolutions following a Bathnorm2d afterwards
the third path contains a 5x5 convolution and normalization
the fourth path is a max pooling path which contains MaxPool2d and a convoution afterwards.

the architecture we ended up using is resnet with batchnorm , bottlenecks. 
we noticed that adding the batchnorm caused a big improvement in accuracy by handling vanishing gradients



2.we reached a higher accuracy on the test set than the accuracy reached in experiment 1, we reached accuracy of around 
78% accuracy. 

*noticed - using a different framed work (pytorch lightning) we reached around 90% accuracy using the inception block 
with a googlenet module, with the following enhancement - scheduler for the learning rate.
"""
# ==============