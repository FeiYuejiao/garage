from garage.tf.algos.npo import NPO
from garage.tf.algos.npo import PGLoss
from garage.tf.optimizers import FirstOrderOptimizer


class VPG(NPO):
    """
    Vanilla Policy Gradient.
    """

    def __init__(self,
                 env_spec,
                 policy,
                 baseline,
                 max_path_length,
                 discount,
                 positive_adv=True,
                 optimizer=None,
                 optimizer_args=None):
        if optimizer is None:
            default_args = dict(
                batch_size=None,
                max_epochs=1,
            )
            optimizer = FirstOrderOptimizer
            if optimizer_args is None:
                optimizer_args = default_args
            else:
                optimizer_args = dict(default_args, **optimizer_args)
        super(VPG, self).__init__(
            env_spec=env_spec,
            policy=policy,
            baseline=baseline,
            max_path_length=max_path_length,
            discount=discount,
            pg_loss=PGLoss.VANILLA,
            positive_adv=positive_adv,
            optimizer=optimizer,
            optimizer_args=optimizer_args,
            name='VPG')
