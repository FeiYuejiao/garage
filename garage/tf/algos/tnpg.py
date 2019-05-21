"""Truncated Natural Policy Gradient."""
from garage.tf.algos.npo import NPO
from garage.tf.optimizers import ConjugateGradientOptimizer


class TNPG(NPO):
    """
    Truncated Natural Policy Gradient.

    TNPG uses Conjugate Gradient to compute the policy gradient.

    """

    def __init__(self,
                 env_spec,
                 policy,
                 baseline,
                 max_path_length,
                 discount,
                 optimizer=None,
                 optimizer_args=None):
        if optimizer is None:
            optimizer = ConjugateGradientOptimizer
            default_args = dict(max_backtracks=1)
            if optimizer_args is None:
                optimizer_args = default_args
            else:
                optimizer_args = dict(default_args, **optimizer_args)
        super().__init__(
            env_spec=env_spec,
            policy=policy,
            baseline=baseline,
            max_path_length=max_path_length,
            discount=discount,
            optimizer=optimizer,
            optimizer_args=optimizer_args,
            name='TNPG')
