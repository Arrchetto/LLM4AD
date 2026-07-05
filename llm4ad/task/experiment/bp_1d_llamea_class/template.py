template_program = '''
class BinPackingOptimizer:
    """Deterministic complete optimizer for one offline 1D BP instance."""

    def __init__(self):
        pass

    def __call__(self, instance: dict) -> dict:
        capacity = instance["bin_capacity"]
        items = instance["items"]
        order = sorted(
            range(len(items)),
            key=lambda index: (-items[index], index),
        )
        bins = []
        remaining = []
        for index in order:
            size = items[index]
            chosen = -1
            least_space = capacity + 1
            for bin_index, space in enumerate(remaining):
                if size <= space and space - size < least_space:
                    chosen = bin_index
                    least_space = space - size
            if chosen < 0:
                bins.append([index + 1])
                remaining.append(capacity - size)
            else:
                bins[chosen].append(index + 1)
                remaining[chosen] -= size
        return {"num_bins": len(bins), "bins": bins}
'''


task_description = (
    "Design a complete optimizer class named BinPackingOptimizer for offline "
    "one-dimensional bin packing. Its no-argument constructor may initialize "
    "any bounded internal search machinery, and __call__(self, instance) must "
    "return a complete feasible packing. The instance dictionary contains "
    "instance_id, bin_capacity, num_items, and items. Return a dictionary with "
    "integer num_bins and bins as a list of nonempty lists. Every bin entry "
    "must be an original 1-based item index, never an item size, and every "
    "index must occur exactly once without exceeding capacity. Duplicate item "
    "sizes are common: preserve identity while reordering and do not use "
    "items.index to recover indices. The class may define multiple member "
    "methods for construction, improvement, local search, or other bounded "
    "procedures. Explore different algorithm families across candidates, not "
    "merely renamed first-fit-decreasing or best-fit-decreasing variants."
)


_namespace: dict[str, object] = {}
exec(template_program, _namespace)
BinPackingOptimizer = _namespace["BinPackingOptimizer"]

__all__ = ["BinPackingOptimizer", "task_description", "template_program"]
