"""
models/scratch_ncf/optimizer.py

SGD with momentum and Adam optimiser — both implemented from scratch.

For thesis, train with Adam (faster convergence) but must
understand SGD first — it is the conceptual foundation.
"""

import numpy as np

class SGD:
    """
    Stochastic Gradient Descent with optional momentum.

    Update rule:
        V <- momentum * V - lr * grad
        W <- W + V
    """

    def __init__(self, params: list[dict], lr: float = 0.01, 
                momentum: float = 0.9):
        self.params = params
        self.lr = lr
        self.momentum = momentum
        # Velocity buffers — one per parameter array
        self.velocities = [np.zeros_like(p["param"]) for p in params]
    
    def step(self):
        """Update all parameters using accumulated gradients."""
        for i, p in enumerate(self.params):
            g = p["grad"]
            self.velocities[i] = (self.momentum * self.velocities[i] 
                                                - self.lr * g)
            p["param"] += self.velocities[i]
    

class Adam:
    """
    Adam optimiser (Kingma & Ba, 2015)

    Maintain per-parameter first moment (m) and 
    second moment (v) estimates.

    Update rule:
      m  <- β1·m + (1−β1)·g          <- biased first moment
      v  <- β2·v + (1−β2)·g²         <- biased second moment
      m̂  = m / (1−β1^t)              <- bias correction
      v̂  = v / (1−β2^t)
      W  <- W − lr · m̂ / (√v̂ + ε)

    Why Adam for NCF?
    Sparse gradients (only embedding rows u and i are updated each step)
    make Adam's per-parameter learning rates particularly effective.
    """

    def __init__(self, params: list[dict], lr: float = 1e-3, 
                beta1 : float = 0.9, beta2: float = 0.999,
                eps: float = 1e-8):
        self.params = params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0                      # time step (for bias correction)
        
        # Initialize first and second moment estimates (m, v) — one per parameter array
        self.ms = [np.zeros_like(p["param"]) for p in params]
        self.vs = [np.zeros_like(p["param"]) for p in params]
    
    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            g = p["grad"]

            # Update biased moment estimates
            self.ms[i] = self.beta1 * self.ms[i] + (1 - self.beta1) * g
            self.vs[i] = self.beta2 * self.vs[i] + (1 - self.beta2) * (g ** 2)
            
            # Compute bias-corrected first and second moments
            m_hat = self.ms[i] / (1 - self.beta1 ** self.t)
            v_hat = self.vs[i] / (1 - self.beta2 ** self.t)
            
            # Update parameters
            p["param"] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)