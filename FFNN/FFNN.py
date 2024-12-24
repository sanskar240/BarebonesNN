import numpy as np

# Define the activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

# Mean Squared Error Loss
def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def mse_loss_derivative(y_true, y_pred):
    return -2 * (y_true - y_pred) / y_true.size

# Initialize weights and biases
class FeedforwardNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.weights1 = np.random.randn(input_size, hidden_size) * 0.01
        self.bias1 = np.zeros((1, hidden_size))
        self.weights2 = np.random.randn(hidden_size, output_size) * 0.01
        self.bias2 = np.zeros((1, output_size))
    
    def forward(self, X):
        # Input to hidden layer
        self.z1 = np.dot(X, self.weights1) + self.bias1
        self.a1 = relu(self.z1) 

        # Hidden to output layer
        self.z2 = np.dot(self.a1, self.weights2) + self.bias2
        self.a2 = sigmoid(self.z2)  

        return self.a2 

    def backward(self, X, y_true, y_pred, learning_rate=0.01):
        # Calculate gradients for the output layer
        loss_derivative = mse_loss_derivative(y_true, y_pred)
        dz2 = loss_derivative * sigmoid_derivative(self.z2)
        dw2 = np.dot(self.a1.T, dz2)
        db2 = np.sum(dz2, axis=0, keepdims=True)

        # Calculate gradients for the hidden layer
        dz1 = np.dot(dz2, self.weights2.T) * relu_derivative(self.z1)
        dw1 = np.dot(X.T, dz1)
        db1 = np.sum(dz1, axis=0, keepdims=True)

        # Update weights and biases
        self.weights1 -= learning_rate * dw1
        self.bias1 -= learning_rate * db1
        self.weights2 -= learning_rate * dw2
        self.bias2 -= learning_rate * db2

    def train(self, X, y, epochs=1000, learning_rate=0.01):
        for epoch in range(epochs):
           
            y_pred = self.forward(X)

            
            loss = mse_loss(y, y_pred)

         
            self.backward(X, y, y_pred, learning_rate)

            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")

# Example usage:
if __name__ == "__main__":
  
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    
    ffnn = FeedforwardNeuralNetwork(input_size=2, hidden_size=4, output_size=1)

   
    ffnn.train(X, y, epochs=1000, learning_rate=0.1)

  
    print("Predictions:")
    print(ffnn.forward(X))
