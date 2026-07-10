def bin_edges(vals: list, n: int) -> list:
    return [vals[int(i * len(vals) / n)] for i in range(1, n)]


def bin_of(val: float, edges: list) -> int:
    for i, e in enumerate(edges):
        if val <= e:
            return i
    return len(edges)


def novelty_from_similarity(nn_sim: float) -> float:
    return 1 - nn_sim
