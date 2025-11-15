
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import make_moons
from IPython.display import clear_output
import time

# -------------------------
# Dataset
# -------------------------
X, y = make_moons(n_samples=500, noise=0.20, random_state=42)
y = y.reshape(-1, 1)
X = (X - X.mean(axis=0)) / X.std(axis=0)

# -------------------------
# Simple MLP
# -------------------------
class SimpleMLP:
    def __init__(self, n_in=2, n_hidden=6, n_out=1, lr=0.1):
        rng = np.random.RandomState(1)
        self.W1 = rng.randn(n_in, n_hidden) * 0.5
        self.b1 = np.zeros((1, n_hidden))
        self.W2 = rng.randn(n_hidden, n_out) * 0.5
        self.b2 = np.zeros((1, n_out))
        self.lr = lr
        self.last_grad_W1 = np.zeros_like(self.W1)
        self.last_grad_W2 = np.zeros_like(self.W2)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_deriv(self, s):
        return s * (1 - s)

    def forward(self, X):
        z1 = X @ self.W1 + self.b1
        a1 = self.sigmoid(z1)
        z2 = a1 @ self.W2 + self.b2
        a2 = self.sigmoid(z2)
        return a2, (X, z1, a1, z2, a2)

    def compute_loss(self, y_hat, y):
        eps = 1e-8
        y_hat = np.clip(y_hat, eps, 1 - eps)
        return -np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))

    def train_step(self, X, y):
        y_hat, cache = self.forward(X)
        Xc, z1, a1, z2, a2 = cache
        m = len(y)
        dz2 = (a2 - y) / m
        dW2 = a1.T @ dz2
        db2 = np.sum(dz2, axis=0, keepdims=True)
        da1 = dz2 @ self.W2.T
        dz1 = da1 * self.sigmoid_deriv(a1)
        dW1 = X.T @ dz1
        db1 = np.sum(dz1, axis=0, keepdims=True)

        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

        self.last_grad_W1 = dW1.copy()
        self.last_grad_W2 = dW2.copy()

        return self.compute_loss(a2, y)

# -------------------------
# Initialize
# -------------------------
model = SimpleMLP()
plt.rcParams["figure.figsize"] = (20, 6)
losses = []
rotation_angle = 0

# -------------------------
# Training Loop
# -------------------------
for epoch in range(301):
    loss = model.train_step(X, y)
    losses.append(loss)

    if epoch % 5 == 0:
        clear_output(wait=True)
        fig = plt.figure(figsize=(20, 6))

        # -------------------------
        # 1) Decision Boundary
        # -------------------------
        ax0 = fig.add_subplot(1, 4, 1)
        xx, yy = np.meshgrid(
            np.linspace(X[:,0].min()-1, X[:,0].max()+1, 200),
            np.linspace(X[:,1].min()-1, X[:,1].max()+1, 200)
        )
        grid = np.c_[xx.ravel(), yy.ravel()]
        Z, _ = model.forward(grid)
        Z = Z.reshape(xx.shape)
        ax0.contourf(xx, yy, Z, levels=50, cmap="coolwarm", alpha=0.8)
        ax0.scatter(X[:,0], X[:,1], c=y[:,0], cmap="coolwarm", edgecolors="k")
        ax0.set_title(f"Decision Boundary\nEpoch {epoch}")
        ax0.set_xlabel("x1")
        ax0.set_ylabel("x2")

        # -------------------------
        # 2) W1 3D Weights
        # -------------------------
        ax1 = fig.add_subplot(1, 4, 2, projection='3d')
        n_in, n_hidden = model.W1.shape
        _x = np.arange(n_hidden)
        _y = np.arange(n_in)
        _xx, _yy = np.meshgrid(_x, _y)
        x, y_bars = _xx.ravel(), _yy.ravel()
        top = model.W1.ravel()
        bottom = np.zeros_like(top)
        width = depth = 0.5
        # Gradient-based coloring for W1
        grad_norm = np.abs(model.last_grad_W1).ravel()
        grad_colors = plt.cm.plasma((grad_norm - grad_norm.min())/(np.ptp(grad_norm)+1e-8))
        ax1.bar3d(x, y_bars, bottom, width, depth, top, shade=True, color=grad_colors)
        rotation_angle += 3
        ax1.view_init(elev=30, azim=rotation_angle)
        ax1.set_xticks(range(n_hidden))
        ax1.set_xticklabels([f"N{i}" for i in range(n_hidden)])
        ax1.set_yticks(range(n_in))
        ax1.set_yticklabels([f"X{i}" for i in range(n_in)])
        ax1.set_title("W1 Weights (Gradient Color)")

        # -------------------------
        # 3) W2 3D Weights
        # -------------------------
        ax2 = fig.add_subplot(1, 4, 3, projection='3d')
        n_hidden, n_out = model.W2.shape
        _x2 = np.arange(n_out)
        _y2 = np.arange(n_hidden)
        _xx2, _yy2 = np.meshgrid(_x2, _y2)
        x2, y2_bars = _xx2.ravel(), _yy2.ravel()
        top2 = model.W2.ravel()
        bottom2 = np.zeros_like(top2)
        width2 = depth2 = 0.5
        grad_norm2 = np.abs(model.last_grad_W2).ravel()
        grad_colors2 = plt.cm.inferno((grad_norm2 - grad_norm2.min())/(np.ptp(grad_norm2)+1e-8))
        ax2.bar3d(x2, y2_bars, bottom2, width2, depth2, top2, shade=True, color=grad_colors2)
        ax2.view_init(elev=30, azim=rotation_angle)
        ax2.set_xticks(range(n_out))
        ax2.set_xticklabels([f"Out{i}" for i in range(n_out)])
        ax2.set_yticks(range(n_hidden))
        ax2.set_yticklabels([f"N{i}" for i in range(n_hidden)])
        ax2.set_title("W2 Weights (Gradient Color)")

        # -------------------------
        # 4) Loss Curve
        # -------------------------
        ax3 = fig.add_subplot(1, 4, 4)
        ax3.plot(losses, label="Loss", color="red")
        ax3.set_title("Training Loss")
        ax3.set_xlabel("Epoch")
        ax3.set_ylabel("Loss")
        ax3.legend()

        plt.tight_layout(rect=[0,0,1,0.95])
        plt.show()
    time.sleep(0.01)
