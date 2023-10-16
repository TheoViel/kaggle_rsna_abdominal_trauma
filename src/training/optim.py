import torch
from collections import defaultdict
from torch.optim.optimizer import Optimizer


class Lookahead(Optimizer):
    """
    Implements the Lookahead optimization algorithm.
    """

    def __init__(self, optimizer, k=5, alpha=0.5):
        """
        Constructor.

        Args:
            optimizer (torch.optim.Optimizer): The base optimizer.
            k (int, optional): Number of steps for lookahead. Defaults to 5.
            alpha (float, optional): Coef for blending the fast & slow weights. Defaults to 0.5.
        """
        self.optimizer = optimizer
        self.k = k
        self.alpha = alpha
        self.param_groups = self.optimizer.param_groups
        self.state = defaultdict(dict)
        self.fast_state = self.optimizer.state
        for group in self.param_groups:
            group["counter"] = 0

    def update(self, group):
        """
        Updates parameters in a group.
        Group represents a parameter group in the optimizer.
        It typically contains the following keys:
          - "params" (a list of parameters),
          - "lr" (learning rate),
          - "momentum",
          - "dampening",
          - "weight_decay",
        and other optimizer-specific parameters.

        Args:
            group (dict): Parameter group.
        """
        for fast in group["params"]:
            param_state = self.state[fast]
            if "slow_param" not in param_state:
                param_state["slow_param"] = torch.zeros_like(fast.data)
                param_state["slow_param"].copy_(fast.data)
            slow = param_state["slow_param"]
            slow += (fast.data - slow) * self.alpha
            fast.data.copy_(slow)

    def update_lookahead(self):
        """
        Updates all parameters.
        """
        for group in self.param_groups:
            self.update(group)

    def step(self, closure=None):
        """
        Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model and returns the loss.

        Returns:
            float: The loss value after the optimization step.
        """
        loss = self.optimizer.step(closure)
        for group in self.param_groups:
            if group["counter"] == 0:
                self.update(group)
            group["counter"] += 1
            if group["counter"] >= self.k:
                group["counter"] = 0
        return loss

    def state_dict(self):
        """
        Returns the state of the optimizer as a dictionary.

        Returns:
            dict: The optimizer state.
        """
        fast_state_dict = self.optimizer.state_dict()
        slow_state = {
            (id(k) if isinstance(k, torch.Tensor) else k): v
            for k, v in self.state.items()
        }
        fast_state = fast_state_dict["state"]
        param_groups = fast_state_dict["param_groups"]
        return {
            "fast_state": fast_state,
            "slow_state": slow_state,
            "param_groups": param_groups,
        }

    def load_state_dict(self, state_dict):
        """
        Loads the optimizer state.

        Args:
            state_dict (dict): The optimizer state dictionary.
        """
        slow_state_dict = {
            "state": state_dict["slow_state"],
            "param_groups": state_dict["param_groups"],
        }
        fast_state_dict = {
            "state": state_dict["fast_state"],
            "param_groups": state_dict["param_groups"],
        }
        super(Lookahead, self).load_state_dict(slow_state_dict)
        self.optimizer.load_state_dict(fast_state_dict)
        self.fast_state = self.optimizer.state

    def add_param_group(self, param_group):
        """
        Adds a parameter group to the optimizer.

        Args:
            param_group (dict): Parameter group.
        """
        param_group["counter"] = 0
        self.optimizer.add_param_group(param_group)


def define_optimizer(
    model, name, lr=1e-3, lr_encoder=1e-3, weight_decay=0, betas=(0.9, 0.999)
):
    """
    Defines an optimizer for the given model based on the specified name.
    Supports discriminative lr between the encoder and other layers.

    Args:
        model (torch.nn.Module): The model for which to define the optimizer.
        name (str): The name of the optimizer.
        lr (float, optional): The learning rate. Defaults to 1e-3.
        lr_encoder (float, optional): The learning rate for encoder layers. Defaults to 1e-3.
        weight_decay (float, optional): The weight decay. Defaults to 0.
        betas (tuple, optional): Optimizer betas. Defaults to (0.9, 0.999).

    Raises:
        NotImplementedError: If the specified optimizer name is not supported.

    Returns:
        torch.optim.Optimizer: The defined optimizer.
    """
    if weight_decay or lr != lr_encoder:
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        opt_params = []
        for n, p in model.named_parameters():
            wd = 0 if any(nd in n for nd in no_decay) else weight_decay
            lr_ = lr_encoder if "encoder" in n else lr
            opt_params.append(
                {
                    "params": [p],
                    "weight_decay": wd,
                    "lr": lr_,
                }
            )
    else:
        opt_params = model.parameters()

    if name.lower() == "ranger":
        radam = getattr(torch.optim, "RAdam")(opt_params, lr=lr, betas=betas)
        return Lookahead(radam, alpha=0.5, k=5)
    try:
        optimizer = getattr(torch.optim, name)(opt_params, lr=lr, betas=betas)
    except AttributeError:
        raise NotImplementedError(name)

    return optimizer
