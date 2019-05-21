from enum import Enum
from enum import unique

from garage.tf.algos.npo import NPO
from garage.tf.algos.npo import PGLoss
from garage.tf.optimizers import ConjugateGradientOptimizer
from garage.tf.optimizers import PenaltyLbfgsOptimizer


@unique
class KLConstraint(Enum):
    HARD = 'hard'
    SOFT = 'soft'


class TRPO(NPO):
    """
    Trust Region Policy Optimization.

    See https://arxiv.org/abs/1502.05477.
    """

    def __init__(self,
                 env_spec,
                 policy,
                 baseline,
                 max_path_length,
                 discount,
                 gae_lambda=0.98,
                 policy_ent_coeff=0.0,
                 max_kl_step=0.01,
                 kl_constraint=KLConstraint.HARD,
                 optimizer=None,
                 optimizer_args=None):

        if not optimizer:
            if kl_constraint == KLConstraint.HARD:
                optimizer = ConjugateGradientOptimizer
            elif kl_constraint == KLConstraint.SOFT:
                optimizer = PenaltyLbfgsOptimizer
            else:
                raise NotImplementedError('Unknown KLConstraint')

        if optimizer_args is None:
            optimizer_args = dict()

        super(TRPO, self).__init__(
            env_spec=env_spec,
            policy=policy,
            baseline=baseline,
            max_path_length=max_path_length,
            discount=discount,
            gae_lambda=gae_lambda,
            pg_loss=PGLoss.SURROGATE,
            optimizer=optimizer,
            optimizer_args=optimizer_args,
            policy_ent_coeff=policy_ent_coeff,
            max_kl_step=max_kl_step,
            name='TRPO')
