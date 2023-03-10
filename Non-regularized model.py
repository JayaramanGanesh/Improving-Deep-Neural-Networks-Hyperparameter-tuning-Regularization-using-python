import matplotlib.pyplot as plt
from reg_utils import  plot_decision_boundary  
from reg_utils import initialize_parameters
from reg_utils import load_2D_dataset
from reg_utils import  predict_dec
from reg_utils import compute_cost 
from reg_utils import predict
from reg_utils import forward_propagation
from reg_utils import backward_propagation
from reg_utils import update_parameters
from testCases import *
from matplotlib import inline


#2d 2d datasets
plt.rcParams['figure.figsize'] = (7.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


#spliting the datasets
train_X, train_Y, test_X, test_Y = load_2D_dataset()


#Non-regularized model
def model(X, Y, learning_rate = 0.3, num_iterations = 30000, print_cost = True, lambd = 0, keep_prob = 1):
    grads = {}
    costs = []   # to keep track of the cost                         
    m = X.shape[1]     # number of examples                   
    layers_dims = [X.shape[0], 20, 3, 1]


    # Initialize parameters dictionary.
    parameters = initialize_parameters(layers_dims)


    for i in range(0, num_iterations):
        # Forward propagation: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID.
        if keep_prob == 1:
            a3, cache = forward_propagation(X, parameters)
        elif keep_prob < 1:
            a3, cache = forward_propagation_with_dropout(X, parameters, keep_prob)
        

        # Cost function
        if lambd == 0:
            cost = compute_cost(a3, Y)
        else:
            cost = compute_cost_with_regularization(a3, Y, parameters, lambd)
            

        # Backward propagation.
        assert(lambd == 0 or keep_prob == 1)    # it is possible to use both L2 regularization and dropout, 
                                            # but this assignment will only explore one at a time
        if lambd == 0 and keep_prob == 1:
            grads = backward_propagation(X, Y, cache)
        elif lambd != 0:
            grads = backward_propagation_with_regularization(X, Y, cache, lambd)
        elif keep_prob < 1:
            grads = backward_propagation_with_dropout(X, Y, cache, keep_prob)
        

        # Update parameters.
        parameters = update_parameters(parameters, grads, learning_rate)
        

        # Print the loss every 10000 iterations
        if print_cost and i % 10000 == 0:
            print("Cost after iteration {}: {}".format(i, cost))
        if print_cost and i % 1000 == 0:
            costs.append(cost)
    

        # plot the cost
        plt.plot(costs)
        plt.ylabel('cost')
        plt.xlabel('iterations (x1,000)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()
     
        return parameters


#plot the accuracy on the train/test sets.
parameters = model(train_X, train_Y)
print("On the training set:")
predictions_train = predict(train_X, train_Y, parameters)
print("On the test set:")
predictions_test = predict(test_X, test_Y, parameters)



#plot the model without regularization
plt.title("Model without regularization")
axes = plt.gca()
axes.set_xlim([-0.75, 0.40])
axes.set_ylim([-0.75, 0.65])
plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)