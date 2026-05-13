import torch
import torch.nn.functional as F


def InfoNCE_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
    tau: float = 0.5,
    normalize: bool = False,
) -> torch.Tensor:
    """
    -(1/N) sum_i log( exp((z1[i] . z2[i]) / tau) /
                      sum_{j != i} exp((z1[i] . z2[j]) / tau) )
    """
    if z1.shape[0] != z2.shape[0]:
        raise ValueError("InfoNCE_loss expects z1 and z2 with the same batch size.")

    if z1.shape[0] < 2:
        raise ValueError("InfoNCE_loss requires at least two samples in the batch.")

    if normalize:
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)

    sim = torch.einsum("id,jd->ij", z1, z2) / tau
    pos = sim.diag()

    neg_mask = ~torch.eye(sim.size(0), dtype=torch.bool, device=sim.device)
    neg_logits = sim.masked_fill(~neg_mask, float("-inf"))
    log_denom = torch.logsumexp(neg_logits, dim=1)

    return -(pos - log_denom).mean()
