"""Stage 2 (session side): sessionisation + a first-order Markov next-item baseline.

* :class:`Sessioniser` cuts each user's timeline into ordered sessions
  (``fixed`` / ``per_user_pct`` / ``single`` strategies).
* :class:`MarkovSessionModel` learns within-session ``(item -> next-item)``
  transitions and scores candidate next videos for the current session.

Companion to :mod:`recsys.user_modelling` (the long-term ``u_long`` model): this
is the short-term, in-session signal that sits *under* ``u_long``, never
overwriting it.
"""

from recsys.session_modelling.markov import MarkovSessionModel
from recsys.session_modelling.sessioniser import Sessioniser

__all__ = ["MarkovSessionModel", "Sessioniser"]
