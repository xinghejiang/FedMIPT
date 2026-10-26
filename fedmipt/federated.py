"""Server-side aggregation (paper Eq. "projector_fedavg")."""

from fedmipt.models import DomainModel


def fedavg_projector(model_a: DomainModel, model_b: DomainModel, count_a: int, count_b: int) -> None:
    """theta <- (n_a theta_a + n_b theta_b) / (n_a + n_b), written back to both domains.

    ``count_*`` are the numbers of local training interactions.  Only the
    semantic projector is aggregated; no other parameter is read by the server.
    """
    if count_a <= 0 or count_b <= 0:
        raise ValueError(f"FedAvg needs positive sample counts, got {count_a} and {count_b}")
    total = count_a + count_b
    state_a = model_a.projector.state_dict()
    state_b = model_b.projector.state_dict()
    averaged = {
        key: state_a[key].detach() * (count_a / total) + state_b[key].detach() * (count_b / total)
        for key in state_a
    }
    model_a.projector.load_state_dict(averaged)
    model_b.projector.load_state_dict(averaged)
