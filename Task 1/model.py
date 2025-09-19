import numpy as np

# --------------------- Layers ---------------------
class Dense:
    """Fully-connected layer"""
    def __init__(self, in_features, out_features, weight_scale=None):
        # Xavier / Glorot initialization
        if weight_scale is None:
            weight_scale = np.sqrt(2.0 / (in_features + out_features))
        self.W = np.random.randn(in_features, out_features) * weight_scale
        self.b = np.zeros((1, out_features))

        # gradients (populated in backward)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

        self._x = None

    def forward(self, x):
        self._x = x
        return x.dot(self.W) + self.b

    def backward(self, grad_out):
        self.dW = self._x.T.dot(grad_out)
        self.db = np.sum(grad_out, axis=0, keepdims=True)
        return grad_out.dot(self.W.T)

    def params_and_grads(self):
        return [(self.W, self.dW), (self.b, self.db)]

# ------------------- Activations -------------------
class ReLU:
    def __init__(self):
        self._mask = None

    def forward(self, x):
        self._mask = (x > 0)
        return x * self._mask

    def backward(self, grad_out):
        return grad_out * self._mask

class Sigmoid:
    def __init__(self):
        self._out = None

    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self._out = out
        return out

    def backward(self, grad_out):
        return grad_out * (self._out * (1 - self._out))

# --------------------- Loss ---------------------
class MSELoss:
    @staticmethod
    def forward(y_pred, y_true):
        diff = y_pred - y_true
        return np.mean(diff * diff)

    @staticmethod
    def backward(y_pred, y_true):
        n = y_true.shape[0]
        return 2.0 * (y_pred - y_true) / n

# ------------------- Neural Network -------------------
class NeuralNetwork:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def backward(self, grad):
        out = grad
        for layer in reversed(self.layers):
            if hasattr(layer, 'backward'):
                out = layer.backward(out)
        return out

    def params_and_grads(self):
        pg = []
        for layer in self.layers:
            if hasattr(layer, 'params_and_grads'):
                pg.extend(layer.params_and_grads())
        return pg

# -------------------- Optimizer --------------------
class SGD:
    """Simple SGD with optional momentum"""
    def __init__(self, lr=1e-3, momentum=0.0):
        self.lr = lr
        self.momentum = momentum
        self._velocities = {}

    def step(self, params_and_grads):
        for (param, grad) in params_and_grads:
            pid = id(param)
            if pid not in self._velocities:
                self._velocities[pid] = np.zeros_like(param)
            v = self._velocities[pid]
            v *= self.momentum
            v -= self.lr * grad
            param += v

# ----------------- Helpers / Builder -----------------
def build_mlp(input_dim, hidden_units, output_dim, activation='relu'):
    activ = ReLU if activation == 'relu' else Sigmoid
    layers = []
    prev = input_dim
    for h in hidden_units:
        layers.append(Dense(prev, h))
        layers.append(activ())
        prev = h
    layers.append(Dense(prev, output_dim))
    return NeuralNetwork(layers)

# -------------- Gradient checking util ---------------
def grad_check(model, x, y, loss_fn=MSELoss, eps=1e-5, tol=1e-4, max_checks_per_param=5):
    y_pred = model.forward(x)
    loss = loss_fn.forward(y_pred, y)
    grad_loss = loss_fn.backward(y_pred, y)
    model.backward(grad_loss)

    for (param, grad) in model.params_and_grads():
        flat_param = param.ravel()
        flat_grad = grad.ravel()
        n_checks = min(max_checks_per_param, flat_param.size)
        rng = np.random.default_rng(123)
        idxs = rng.choice(flat_param.size, size=n_checks, replace=False)

        for idx in idxs:
            orig = flat_param[idx]
            flat_param[idx] = orig + eps
            loss_p = loss_fn.forward(model.forward(x), y)
            flat_param[idx] = orig - eps
            loss_m = loss_fn.forward(model.forward(x), y)
            flat_param[idx] = orig

            numerical = (loss_p - loss_m) / (2 * eps)
            analytical = flat_grad[idx]
            denom = max(1e-8, abs(numerical) + abs(analytical))
            rel_error = abs(numerical - analytical) / denom
            if rel_error > tol:
                print(f"Grad check FAILED at index {idx}: num={numerical:.6e}, ana={analytical:.6e}, rel_err={rel_error:.3e}")
                return False
    print("Grad check passed (sampled checks).")
    return True

if __name__ == '__main__':
    np.random.seed(0)
    X = np.random.randn(10, 1)
    y = (0.5 * X**3 - 2 * X**2 + X) + 0.1 * np.random.randn(10, 1)
    net = build_mlp(1, [16, 16], 1)
    loss = MSELoss.forward(net.forward(X), y)
    print('Initial loss:', loss)
