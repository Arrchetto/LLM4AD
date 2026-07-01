template_program = '''
def solve(instance_id: str, bin_capacity: int, num_items: int, items: list[int]) -> dict:
    """Construct a complete feasible packing for one offline 1D BP instance."""
    order = sorted(range(num_items), key=lambda index: items[index], reverse=True)
    bins = []
    remaining = []
    for index in order:
        size = items[index]
        chosen = -1
        least_space = bin_capacity + 1
        for bin_index, space in enumerate(remaining):
            if size <= space and space - size < least_space:
                chosen = bin_index
                least_space = space - size
        if chosen < 0:
            bins.append([index + 1])
            remaining.append(bin_capacity - size)
        else:
            bins[chosen].append(index + 1)
            remaining[chosen] -= size
    return {"num_bins": len(bins), "bins": bins}
'''


task_description = (
    "Design a complete algorithm for offline one-dimensional bin packing. "
    "The solve function receives one full instance and must assign every item "
    "exactly once to capacity-feasible bins using one-based item indices. "
    "Return a dictionary containing num_bins and bins. Minimize the number of "
    "bins. Implement the whole solver inside solve, including construction and "
    "any improvement phases; do not return a priority score or partial rule. "
    "The training set contains 30 BPPLIB instances with about 100 items each, "
    "so keep the algorithm deterministic and computationally bounded."
)


def solve(instance_id: str, bin_capacity: int, num_items: int, items: list[int]) -> dict:
    """Baseline best-fit-decreasing solver used by tests and documentation."""
    namespace: dict[str, object] = {}
    exec(template_program, namespace)
    return namespace["solve"](instance_id, bin_capacity, num_items, items)

