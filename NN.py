import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

class MLP:
    def __init__(self, X, Y, learning_rate, inLayer_neurons, hiddenLayer_neurons, output_neurons, epochs):
        self.learning_rate = learning_rate
        self.X = X
        self.Y = Y
        self.inLayer_neurons = inLayer_neurons
        self.hiddenLayer_neurons = hiddenLayer_neurons
        self.output_neurons = output_neurons
        self.epochs = epochs


    def initializeWB(self):
        wh = np.random.uniform(size=(self.inLayer_neurons, self.hiddenLayer_neurons))
        bh = np.random.uniform(size=(1, self.hiddenLayer_neurons))
        wout = np.random.uniform(size=(self.hiddenLayer_neurons, self.output_neurons))
        bout = np.random.uniform(size=(1, self.output_neurons))

        return wh, wout, bh, bout


    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))


    def derivative_sigmoid(self, x):
        return x * (1 - x)


    def forward_propagation(self, wh, wout, bh, bout):
        hidden_layer_input = np.dot(self.X, wh) + bh
        hidden_layer_activation = self.sigmoid(hidden_layer_input)
        output_layer_input = np.dot(hidden_layer_activation, wout) + bout
        output = self.sigmoid(output_layer_input)

        return hidden_layer_input, hidden_layer_activation, output_layer_input, output


    def backward_propagation(self, output, hidden_layer_activation, wh, wout, bh, bout):
        E = self.Y - output
        slope_output_layer = self.derivative_sigmoid(output)
        slope_hidden_layer = self.derivative_sigmoid(hidden_layer_activation)
        d_output = E * slope_output_layer
        Error_at_hidden_layer = d_output.dot(wout.T)
        d_hidden_layer = Error_at_hidden_layer * slope_hidden_layer
        wout += hidden_layer_activation.T.dot(d_output) * self.learning_rate
        bout += np.sum(d_output, axis=0, keepdims=True) * self.learning_rate
        wh += self.X.T.dot(d_hidden_layer) * self.learning_rate
        bh += np.sum(d_hidden_layer, axis=0, keepdims=True) * self.learning_rate

        return wh, wout, bh, bout


    def compute_loss(self, output):
        return np.mean((self.Y - output) ** 2)

    
    def train(self):
        wh, wout, bh, bout = self.initializeWB()
        loss_history = []

        for i in range(self.epochs):
            hli, hla, oli, output = self.forward_propagation(wh, wout, bh, bout)
            wh, wout, bh, bout = self.backward_propagation(output, hla, wh, wout, bh, bout)
            
            loss = self.compute_loss(output)

            loss_history.append(loss)
            
            if i % 100 == 0:
                print(f"Epoch {i}, Loss: {loss}")

        self.plot_graphs(loss_history)
        self.plot_decision_boundary(wh,wout,output)



    def plot_graphs(self, loss_history):
        plt.figure(figsize=(6, 6))
        plt.plot(loss_history, label='Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss over Epochs')
        plt.legend()

        
        plt.tight_layout()
        plt.show()

    def plot_decision_boundary(self,wh,wout,output):
        steps = 1000
        x_span = np.linspace(self.X[:, 0].min(), self.X[:, 0].max(), steps)
        y_span = np.linspace(self.X[:, 1].min(), self.X[:, 1].max(), steps)
        xx, yy = np.meshgrid(x_span, y_span)

        # Forward pass for region of interest
        hiddenLayer_linearTransform = np.dot(wh.T, np.c_[xx.ravel(), yy.ravel()].T)
        hiddenLayer_activations = self.sigmoid(hiddenLayer_linearTransform)
        outputLayer_linearTransform = np.dot(wout.T, hiddenLayer_activations)
        output_span = self.sigmoid(outputLayer_linearTransform)

        # Make predictions across region of interest
        labels = (output_span > 0.5).astype(int)

        # Plot decision boundary in region of interest
        plt.figure(figsize=(8, 6))
        plt.contourf(xx, yy, labels.reshape(xx.shape), alpha=0.2)
        plt.scatter(self.X[:, 0], self.X[:, 1], s=10, c=self.Y.flatten(), cmap='coolwarm', label='Data')
        plt.scatter(self.X[:, 0], self.X[:, 1], c=output.flatten() > 0.5, cmap='coolwarm', marker='x', label='Predicted')
        plt.title('Decision Boundary')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.legend()
        plt.show()

# Generate data
X, y = make_blobs(n_samples=300, centers=2, n_features=2, random_state=42)

# Reshape y to be a column vector
y = y.reshape(-1, 1)

# Initialize and train the MLP
obj = MLP(X, y, learning_rate=0.001, inLayer_neurons=X.shape[1], hiddenLayer_neurons=3, output_neurons=1, epochs=5000)
obj.train()
