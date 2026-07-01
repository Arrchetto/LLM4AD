template_program = '''
def solve(instance_id: str, bin_capacity: int, num_items: int, items: list[int]) -> dict:
    """Construct a complete feasible packing for one offline 1D BP instance.

    Return a dictionary with exactly two entries: integer ``num_bins`` and
    ``bins`` as ``list[list[int]]``. ``bins`` must be a list of lists, never a
    dictionary. Each inner list contains the original 1-based item indices
    assigned to that bin, not item sizes. Every index from 1 through
    ``num_items`` must occur exactly once, and each bin's item-size sum must
    not exceed ``bin_capacity``.

    Duplicate sizes are common. Preserve item identity while reordering, for
    example with ``(index + 1, items[index])`` pairs. Do not use items.index
    to recover indices and do not return sorted item values as assignments.
    """
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
    "exactly once to capacity-feasible bins using its original 1-based item "
    "index. Return exactly a dictionary containing integer num_bins and bins "
    "as a list of lists of indices; bins must never be a dictionary or contain "
    "item sizes. Duplicate sizes are common, so preserve identity with pairs "
    "such as (index + 1, items[index]); do not use items.index to reconstruct "
    "indices. Minimize the number of bins. Implement the whole solver inside "
    "solve, including construction and any improvement phases; do not return "
    "a priority score or partial rule. "
    "Explore genuinely different algorithm families across candidates, not "
    "merely renamed variants of first-fit or best-fit decreasing. "
    "The training set contains 30 BPPLIB instances with about 100 items each, "
    "so keep the algorithm deterministic and computationally bounded."
)


def solve(instance_id: str, bin_capacity: int, num_items: int, items: list[int]) -> dict:
    """Baseline best-fit-decreasing solver used by tests and documentation."""
    namespace: dict[str, object] = {}
    exec(template_program, namespace)
    return namespace["solve"](instance_id, bin_capacity, num_items, items)
