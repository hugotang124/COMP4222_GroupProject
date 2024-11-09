import torch
import torch.optim as optim

class Optim(object):
    def __init__(self, params, method, lr, clip, lr_decay=1.0, start_decay_at=None):
        """
        Initialize the optimizer.

        Args:
            params: Parameters to optimize (can be a generator).
            method (str): Optimization method (e.g., 'sgd', 'adam').
            lr (float): Learning rate.
            clip (float): Gradient clipping value.
            lr_decay (float): Factor by which to decay the learning rate.
            start_decay_at (int, optional): Epoch at which to start decaying the learning rate.
        """
        self.params = params  
        self.lr = lr
        self.clip = clip
        self.method = method
        self.lr_decay = lr_decay
        self.start_decay_at = start_decay_at
        self.last_ppl = None
        self.start_decay = False

        self.optimizer = self._make_optimizer()

    def _make_optimizer(self):
        """Create the optimizer based on the specified method."""
        optimizer_methods = {
            'sgd': optim.SGD,
            'adagrad': optim.Adagrad,
            'adadelta': optim.Adadelta,
            'adam': optim.Adam
        }

        if self.method not in optimizer_methods:
            raise RuntimeError(f"Invalid optimization method: {self.method}")

        return optimizer_methods[self.method](self.params, lr=self.lr, weight_decay=self.lr_decay)

    def step(self):
        """Perform a single optimization step."""
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.params, self.clip)

        # for param in self.params:
        #     grad_norm += math.pow(param.grad.data.norm(), 2)
        #
        # grad_norm = math.sqrt(grad_norm)
        # if grad_norm > 0:
        #     shrinkage = self.max_grad_norm / grad_norm
        # else:
        #     shrinkage = 1.
        #
        # for param in self.params:
        #     if shrinkage < 1:
        #         param.grad.data.mul_(shrinkage)
        
        self.optimizer.step()
        return self.optimizer.state.get('grad_norm', 0)  # Placeholder for gradient norm if needed

    def update_learning_rate(self, ppl, epoch):
        """
        Update the learning rate based on validation performance.

        Args:
            ppl (float): Perplexity or validation performance metric.
            epoch (int): Current epoch number.
        """
        if self.start_decay_at is not None and epoch >= self.start_decay_at:
            self.start_decay = True
        if self.last_ppl is not None and ppl > self.last_ppl:
            self.start_decay = True

        if self.start_decay:
            self.lr *= self.lr_decay
            print(f"Decaying learning rate to {self.lr:.6g}")

            # Recreate the optimizer with the updated learning rate
            self.optimizer = self._make_optimizer()

        self.last_ppl = ppl
        self.start_decay = False  # Reset after one decay
        #test push