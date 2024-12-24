import numpy as np

def conv2d(X, kernel, stride=1, padding=0):
    """
    Perform a 2D convolution.
    X: Input matrix (image or feature map)
    kernel: Convolution kernel (filter)
    stride: Step size for sliding the filter
    padding: Number of zeros added around the border
    """
   
    if padding > 0:
        X = np.pad(X, ((padding, padding), (padding, padding)), mode='constant')

    h, w = X.shape
    kh, kw = kernel.shape

  
    out_h = (h - kh) // stride + 1
    out_w = (w - kw) // stride + 1

    output = np.zeros((out_h, out_w))

    for i in range(out_h):
        for j in range(out_w):
            region = X[i * stride:i * stride + kh, j * stride:j * stride + kw]
            output[i, j] = np.sum(region * kernel)

    return output

# Example convolution
image = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])
kernel = np.array([
    [1, 0],
    [0, -1]
])
conv_result = conv2d(image, kernel)
print(conv_result)

def relu(X):
    return np.maximum(0, X)

# Example
relu_result = relu(conv_result)
print(relu_result)

def max_pooling(X, pool_size=2, stride=2):
    h, w = X.shape
    out_h = (h - pool_size) // stride + 1
    out_w = (w - pool_size) // stride + 1

    output = np.zeros((out_h, out_w))
    for i in range(out_h):
        for j in range(out_w):
            region = X[i * stride:i * stride + pool_size, j * stride:j * stride + pool_size]
            output[i, j] = np.max(region)

    return output


pooled_result = max_pooling(relu_result)
print(pooled_result)

def fully_connected(X, weights, bias):
    return np.dot(X, weights) + bias

# Example fully connected
fc_weights = np.array([[0.1, 0.2], [0.3, 0.4]])
fc_bias = np.array([0.5, 0.6])
flattened_input = pooled_result.flatten()
fc_output = fully_connected(flattened_input, fc_weights, fc_bias)
print(fc_output)

def softmax(X):
    exps = np.exp(X - np.max(X))  # Stability trick
    return exps / np.sum(exps)

# Example softmax
probabilities = softmax(fc_output)
print(probabilities)


