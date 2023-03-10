**1 Foundations of Deep Learning**

artificial intelligence, machine learning, deep learning
	teaching computers how to learn a task directly from raw data

Why deep learning and why not?
	underlying features
	low level (lines, edges)
	mid level (eyes, nose, ears)
	high level (facial structure)

Why now?: Big data, hardware, software

The perceptron: the structural building block of deep learning
	forward propagation:
	inputs, weights, sum, non-linearity, output
	y = g (w0 + ∑xi•wi)
	y = g (w0 + X(transpose)W)
	w0 - bias, g - sigmoid function (1 / (1 + e ^ (-z)))
	other functions inlcdue Hyperbolic tangent and Rectified Linear Unit (ReLU)

Importance of Activation Functions:
	introduce non-linearities into the network

Building Neural Networks with Perceptrons
	Multi Output Perceptron: dense layers
	Inputs, hidden, outputs
	deep neural network: stacking layers

Applying neural networks
	example problem: will I pass this class
	inputs: hours spent on the final project, number of lectures you attend
	quantifying loss:
		the loss of our network measures the total loss over our entire dataset
		the loss (empirical) function: L(f(x^(i); W), y^(i)) - predicted and actual
		J(W) = 1 / n ∑ (loss function)
	cross entropy loss:
		binary cross entropy loss:
			J(W) = -1 / n * ∑ (y^(i) * log(f(x^i, W)) + (1 - y^(i)) * log(1 - f(x^i, W))))
		final score: mean square error loss
			J(W) = 1 / n * ∑ ( y^(i) - f(x^i, W)) ^ 2

Training Neural Networks
	the network weights that achieve the lowest loss
		W* = argmin 1/n ∑ (loss function for W)
		W* = argmin J(W)
		our loss is a function of the network weights
	loss optimization
		compute gradient, to reduce loss, until convergence
		Gradient Descent, algorithm
			initialize weights randomly
			loop until convergence
			compute gradient
			update weights (how much trust, how much step in)
			return weights
	Computing Gradients: Backpropogation

Neural Networks in Practice: Optimization
	difficult
	optimization through gradient descent
	W - n ∂J(W) / ∂W (n - learning rate, how to set?)
		too low - converges slowly and gets stuck in false local minima
		too high - overshoot, become unstable and diverge
	How to pick?
		smart idea: design an adaptive learning rate that "adapts" to the landscape
	Gradient Descent Algorithms:
		SGD, Adam, Adadelta, Adagrad, RMSProp

Neural Networks in Practice: Mini-batches
	gradient - computationally intensive
	gradient over one Ji: easy to compute, but very stochastic
	middle ground: compute a mini batch of examples (fast to compute and a much better estimate of the true gradient)
	fast training, parallelize computation - speed increase on GPU

Neural Networks in Practice: Overfitting
	Too complex, extra parameters, does not generalize well
	Regularization
		technique that constrains our optimization problem to discourage complex models
		improve generalization of our model on unseen data
	Method 1. Dropout
		During training, randomly set some activations to 0
		typically drop 50%
		force network to not rely on any single node
	Method 2. Early Stopping
		Stop training before we have a chance to overfit
		loss on testing set first decreases then increases after a point as training iterations increase

Core Foundation Review
	The perceptron
		structual building blocks
		nonlinear activation functions
	Neural Networks
		stacking perceptrons to form neural networks
		optimization through backpropogation
	Training in practice
		adaptive learning
		batching
		regularization