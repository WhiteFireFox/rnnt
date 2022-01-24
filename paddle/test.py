import paddle
import unittest
import numpy as np
import rnnt_ops as core


xs = paddle.to_tensor([], dtype=paddle.float32)
ys = paddle.to_tensor([], dtype=paddle.int32)
xn = paddle.to_tensor([], dtype=paddle.int32)
yn = paddle.to_tensor([], dtype=paddle.int32)


class RNNTLossTest(unittest.TestCase):
    def test_calls(self):

        n = 128
        t = 100
        u = 90
        v = 3

        for i in range(2):

          rng = np.random.RandomState(i)

          xs = rng.randn(n, t, u, v)
          xs = paddle.to_tensor(xs, dtype=paddle.float32)
          xs = paddle.nn.functional.log_softmax(xs, axis=-1)

          ys = paddle.to_tensor(rng.randint(1, v, (n, u-1)), dtype=paddle.int32)

          xn = paddle.to_tensor([t] * n, dtype=paddle.int32)
          yn = paddle.to_tensor(rng.randint(1, u, n), dtype=paddle.int32)

          costs, grads = core.rnnt_loss(xs, ys, xn, yn)

if __name__ == "__main__":
    unittest.main()
