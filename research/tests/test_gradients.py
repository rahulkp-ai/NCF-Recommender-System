# tests/test_gradients.py
# NCF-Recommender-System % python -m research.tests.test_gradients
import sys

sys.path.append(".")
from research.models.scratch_ncf.forward import ScratchNCF
from research.models.scratch_ncf.loss import BCELoss


def numerical_gradient(model, loss_fn, uid, iid, y, param, idx, eps=1e-5):
    """Compute numerical gradient for param[idx] using central differences."""
    original = param.flat[idx]

    param.flat[idx] = original + eps
    loss_plus = loss_fn.forward(model.forward(uid, iid), y)
    loss_fn.zero_grad()

    param.flat[idx] = original - eps
    loss_minus = loss_fn.forward(model.forward(uid, iid), y)
    loss_fn.zero_grad()

    param.flat[idx] = original
    return (loss_plus - loss_minus) / (2 * eps)


def test_output_layer_gradient():
    model = ScratchNCF(n_users=10, n_items=10, emb_dim=4, hidden=[8, 4])
    loss_fn = BCELoss()

    uid, iid, y = 3, 7, 1.0
    logit = model.forward(uid, iid)
    loss_fn.forward(logit, y)
    model.backward(loss_fn.backward())

    # Check one weight in the output linear layer
    W_out = model.linears[-1].W
    dW_analytic = model.linears[-1].dW[0, 0]
    dW_numeric = numerical_gradient(model, loss_fn, uid, iid, y, W_out, 0)

    rel_err = abs(dW_analytic - dW_numeric) / max(abs(dW_analytic), abs(dW_numeric), 1e-8)
    assert rel_err < 1e-4, f"Gradient check FAILED: rel_err={rel_err:.2e}"
    print(f"Gradient check PASSED: rel_err={rel_err:.2e}")


if __name__ == "__main__":
    test_output_layer_gradient()
