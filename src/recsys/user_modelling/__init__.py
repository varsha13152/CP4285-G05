"""Stage 2: long-term user modelling (``u_long``) via matrix factorization.

Exposes :class:`UserModeller`, which trains an implicit-feedback ALS model over
KuaiRec's ``big_matrix`` and produces an :class:`~recsys.contracts.models.EngagementProfile`
(carrying the learned latent ``user_vector``) for downstream candidate generation
and ranking.
"""

from recsys.user_modelling.user_modeller import UserModeller

__all__ = ["UserModeller"]
