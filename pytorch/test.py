import torch
import unittest
import numpy as np
import warp_rnnt._C as core


xs = torch.tensor([], dtype=torch.float32)
ys = torch.tensor([], dtype=torch.int)
xn = torch.tensor([], dtype=torch.int)
yn = torch.tensor([], dtype=torch.int)


class RNNTLossTest(unittest.TestCase):
    def test_calls(self):

        n = 128
        t = 100
        u = 90
        v = 3

        for i in range(2):

          rng = np.random.RandomState(i)

          xs = rng.randn(n, t, u, v)
          xs = torch.tensor(xs, dtype=torch.float32)
          xs = torch.nn.functional.log_softmax(xs, dim=-1)

          ys = torch.tensor(rng.randint(1, v, (n, u-1)), dtype=torch.int)

          xn = torch.tensor([t] * n, dtype=torch.int)
          yn = torch.tensor(rng.randint(1, u, n), dtype=torch.int)

          costs, grads = core.rnnt_loss(
            xs.cuda(), ys.cuda(),
            xn.cuda(), yn.cuda())

if __name__ == "__main__":
    unittest.main()
