"""Stage 2 (session side): sessionisation + a first-order Markov next-item baseline.

* :class:`Sessioniser` cuts each user's timeline into ordered sessions
  (``fixed`` / ``per_user_pct`` / ``single`` strategies).
* :class:`MarkovSessionModel` learns within-session ``(item -> next-item)``
  transitions and scores candidate next videos for the current session.
* :class:`SessionModeller` trains both on construction and exposes the pieces the
  pipeline consumes — ``predict_next`` (the SESSION retrieval lane) and the real
  per-user ``avg_session_length`` for ``EngagementProfile``.

Companion to :mod:`recsys.user_modelling` (the long-term ``u_long`` model): this
is the short-term, in-session signal that sits *under* ``u_long``, never
overwriting it.
"""

from recsys.session_modelling.markov import MarkovSessionModel
from recsys.session_modelling.session_modeller import SessionModeller
from recsys.session_modelling.sessioniser import Sessioniser

__all__ = ["MarkovSessionModel", "SessionModeller", "Sessioniser"]
