"""Internal observation state; never a public success or an ownership transfer."""

class SpawnObservationPending(RuntimeError):
    """RSS/headroom sampled, but Popen has not yet returned owned registration.

    Only a watchdog may defer classification for this bounded transition. Result
    publication and owner finalization must still obtain a complete observation.
    """
    def __init__(self, observed):
        super().__init__('Owned spawn registration observation pending')
        self.observed = observed
