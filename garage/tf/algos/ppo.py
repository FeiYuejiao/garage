"""This module implements a PPO algorithm."""
from garage.tf.algos.npo import NPO
from garage.tf.algos.npo import PGLoss
from garage.tf.optimizers import FirstOrderOptimizer


class PPO(NPO):
    """
    Proximal Policy Optimization.

    See https://arxiv.org/abs/1707.06347.
    """

    def __init__(self,
                 env_spec,
                 policy,
                 baseline,
                 max_path_length,
                 discount,
                 lr_clip_range,
                 plot=False,
                 optimizer=None,
                 optimizer_args=None):
        """
        Construct class.

        Args:
            env_spec (garage.envs.EnvSpec): Environment specification.
            policy (garage.tf.policies.base.Policy): Policy.
            baseline (garage.tf.baselines.Baseline): The baseline.
            max_path_length (int): Maximum length of a single rollout.
            discount (float): Discount.
            gae_lambda (float): Lambda used for generalized advantage
                estimation.
            optimizer (float): The optimizer of the algorithm.
            optimizer_args (dict): Optimizer args.
        """
        if optimizer is None:
            optimizer = FirstOrderOptimizer
            if optimizer_args is None:
                optimizer_args = dict()
        super(PPO, self).__init__(
            env_spec=env_spec,
            policy=policy,
            baseline=baseline,
            max_path_length=max_path_length,
            discount=discount,
            lr_clip_range=lr_clip_range,
            pg_loss=PGLoss.SURROGATE_CLIP,
            optimizer=optimizer,
            optimizer_args=optimizer_args,
            name='PPO')
