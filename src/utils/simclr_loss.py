import torch
import torch.nn as nn
import torch.nn.functional as F


def calc_simclr_loss(
    z: torch.Tensor,
    n: int,
    similarity: str = "cosine",
    cosine_temperature: int = 1,
):
    if similarity == "cosine":
        similarity_f = nn.CosineSimilarity(dim=2)
        similarity_matrix = (
            similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / cosine_temperature
        )
    elif similarity == "dot":
        similarity_matrix = z @ z.t()

    sim_i_j = torch.diag(similarity_matrix, n)
    sim_j_i = torch.diag(similarity_matrix, -n)

    positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).view(2 * n, 1)

    mask = torch.ones(2 * n, 2 * n, dtype=torch.bool, device=z.device).fill_diagonal_(0)

    negative_samples = similarity_matrix[mask].view(2 * n, -1)

    labels = torch.zeros(2 * n, dtype=torch.long, device=z.device)
    logits = torch.cat((positive_samples, negative_samples), dim=1)

    simclr_loss = F.cross_entropy(logits, labels, reduction="none") / 2
    acc = (torch.argmax(logits, dim=1) == labels).float()

    return simclr_loss, acc
