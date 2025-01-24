import torch
from time import time


def is_transitive(adjacency: torch.tensor) -> bool:
    return adjacency[torch.matmul(adjacency, adjacency) > 0].all()


def greedy_additive(costs: torch.Tensor, adjacency: torch.Tensor = None) -> (torch.Tensor, torch.Tensor):
    """
    Algorithm 2 from the paper.

    This algorithm for the preorder problem starts with a reflexive but otherwise empty binary relation.
    Iteratively an arc (i,j) is chosen such that adding (i,j) as well as all transitive implications to the relation
    increases the total cost maximally. The algorithm terminates if adding any arc reduces the total cost.

    More specifically, let $x \in \{0,1\}^{n \times n}$ be the $n \times n$ adjacency-matrix with $x_{ij} = 1$ iff $ij$
    is in the relation (initially $x_{ij} = 1$ iff $i == j$). For $ij$ with $x_{ij} = 0$, setting $x_{ij}$ to 1 implies
    that $x_{kl} = 1$ for all $k$ with $x_{ki} = 1$ and all $l$ with $x_{jl} = 1$. The gain of total cost by setting
    $x_{ij} = 1$ is thus given by
    \[
        g_{ij} = \sum_k \sum_l x_{ki} x_{jl} c_{kl} (1-x_{kl})
    \]
    The matrix $g$ can be computed by a simple matrix multiplication:
    \[
        g = x^\top (c \odot (1 - x)) x^\top
    \]
    where $\odot$ is the element-wise matrix multiplication.

    :param costs: n by n cost matrix
    :param adjacency: optional starting point
    :return: total cost of computed transitive relation and adjacency matrix
    """
    assert costs.ndim == 2
    assert costs.shape[0] == costs.shape[1]
    n = costs.shape[0]

    # start with a reflexive but otherwise empty relation
    if adjacency is None:
        adjacency = torch.eye(n, dtype=costs.dtype, device=costs.device)
    else:
        assert is_transitive(adjacency)

    total_cost = (costs * adjacency).sum()

    while True:
        # pick arc i->j such that adding it and all its transitive implications
        # to the relation increases the total cost maximally
        gains = torch.matmul(torch.matmul(adjacency.T, costs * (1-adjacency)), adjacency.T)
        gains -= adjacency
        # subtract the adjacency such that all arcs that are already in the relation have negative gain
        idx = torch.argmax(gains)
        i = idx // n
        j = idx % n

        # terminate if gain is negative
        if gains[i, j] < 0:
            break
        assert adjacency[i, j] == 0

        # update total cost
        total_cost += gains[i, j]
        # update adjacency
        adjacency[torch.outer(adjacency[:, i] > 0, adjacency[j, :] > 0)] = 1

        # assert (adjacency * costs).sum() == total_cost
        # assert is_transitive(adjacency)

    return total_cost, adjacency


def test():
    n = 1000
    device = torch.device('cuda')

    torch.random.manual_seed(0)
    costs = torch.randint(-1, 2, (n, n)).float().to(device)
    # NOTE: It is important to use floats instead of double otherwise GPU speedup is small!

    t_0 = time()
    total_cost, adjacency = greedy_additive(costs)
    print(time() - t_0)
    print(total_cost.item())


if __name__ == '__main__':
    test()
