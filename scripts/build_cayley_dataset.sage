"""
build_cayley_dataset.sage

Run with:  sage build_cayley_dataset.sage

Generates a dataset of edge-labelled Cayley graphs for finite groups,
with binary labels (abelian / non-abelian), saved as JSON for easy
ingestion into PyTorch Geometric.

Output: cayley_dataset.json
Schema per entry:
  {
    "group_id"    : str,          # e.g. "SmallGroup(16,2)"
    "order"       : int,
    "is_abelian"  : bool,
    "generators"  : [str, ...],   # generator labels
    "nodes"       : [int, ...],   # 0-indexed node ids
    "edges"       : [[src, dst, gen_idx], ...],
    "node_labels" : [str, ...],   # human-readable element names
  }
"""

import json
import sys
from itertools import combinations

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Which orders to include. Keep ≤ 64 for reasonable graph sizes.
ORDERS = list(range(2, 33))          # orders 2..32 — adjust as needed

# How many generating sets to sample per group.
# Set to 1 for the canonical minimal generating set only.
# Set to 3+ for generating-set-augmented diversity.
SAMPLES_PER_GROUP = 2

# Maximum generating set size to try.
MAX_GEN_SET_SIZE = 4

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def group_id_str(G):
    """Return a canonical string id for a group G."""
    try:
        n = G.order()
        idx = gap.SmallGroupsInformation(n)   # just to verify GAP database exists
        gid = G.gap().IdGroup()
        return "SmallGroup({},{})".format(gid[0], gid[1])
    except Exception:
        return "{}".format(G.structure_description())


def find_generating_sets(G, n_samples=1, max_size=MAX_GEN_SET_SIZE):
    """
    Return up to n_samples distinct generating sets for G.
    Strategy:
      1. Always include the canonical GAP-minimal generating set.
      2. Sample additional ones by random selection from G.
    """
    elems = list(G)
    identity = G.identity()
    results = []

    # --- Canonical minimal generating set via GAP ---
    try:
        gap_gens = list(G.gap().MinimalGeneratingSet())
        canonical = [G(g) for g in gap_gens]
        if canonical and G.is_generated_by(canonical):
            results.append(canonical)
    except Exception:
        pass

    # --- Random samples ---
    import random
    random.seed(42)
    attempts = 0
    while len(results) < n_samples and attempts < 200:
        attempts += 1
        size = random.randint(1, min(max_size, len(elems) - 1))
        candidates = [e for e in elems if e != identity]
        if len(candidates) < size:
            break
        sample = random.sample(candidates, size)
        if G.is_generated_by(sample):
            # Deduplicate by frozenset of element orders and positions
            key = frozenset(s.__repr__() for s in sample)
            if not any(frozenset(r.__repr__() for r in existing) == key
                       for existing in results):
                results.append(sample)

    if not results:
        # Fallback: use all non-identity elements
        non_id = [e for e in elems if e != identity]
        if G.is_generated_by(non_id):
            results.append(non_id)

    return results[:n_samples]


def cayley_graph_to_dict(G, gens, group_id, order, is_abelian):
    """
    Build a Cayley graph Cay(G, S ∪ S⁻¹) and serialise it.

    Nodes  : group elements (0-indexed by enumeration order)
    Edges  : directed; one edge per (element, generator) pair
             target = element * generator
    Edge feature: generator index (int) in the gens list
    """
    elems = list(G)
    elem_to_idx = {e: i for i, e in enumerate(elems)}

    # Expand generating set to include inverses (standard for undirected Cayley)
    expanded_gens = list(gens)
    for g in list(gens):
        inv = g ** -1
        if inv not in expanded_gens:
            expanded_gens.append(inv)

    gen_labels = [str(g) for g in expanded_gens]

    edges = []
    for i, elem in enumerate(elems):
        for gen_idx, gen in enumerate(expanded_gens):
            target = elem * gen
            j = elem_to_idx[target]
            edges.append([i, j, gen_idx])

    return {
        "group_id"   : group_id,
        "order"      : order,
        "is_abelian" : is_abelian,
        "generators" : gen_labels,
        "nodes"      : list(range(len(elems))),
        "edges"      : edges,
        "node_labels": [str(e) for e in elems],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    dataset = []
    seen_configs = set()   # (group_id, frozenset of gen reprs) — avoid duplicates

    print("Building Cayley graph dataset...")
    print("Orders: {}..{}".format(min(ORDERS), max(ORDERS)))
    print("Samples per group: {}".format(SAMPLES_PER_GROUP))
    print("-" * 50)

    total_groups = 0
    total_graphs = 0

    for order in ORDERS:
        # Enumerate all groups of this order via GAP's SmallGroups library
        try:
            n_groups = number_of_small_groups(order)
        except Exception:
            print("  Order {}: SmallGroups database not available, skipping.".format(order))
            continue

        for i in range(1, n_groups + 1):
            try:
                G = SmallPermutationGroup(order, i)
            except Exception:
                try:
                    G = gap.SmallGroup(order, i)
                    G = G.AsPermGroup()    # convert to Sage permutation group
                    G = PermutationGroup(G.GeneratorsOfGroup())
                except Exception as e:
                    print("  Could not load SmallGroup({},{}): {}".format(order, i, e))
                    continue

            gid = group_id_str(G)
            is_ab = G.is_abelian()
            total_groups += 1

            gen_sets = find_generating_sets(G, n_samples=SAMPLES_PER_GROUP)
            if not gen_sets:
                print("  {}: could not find generating set, skipping.".format(gid))
                continue

            for gens in gen_sets:
                key = (gid, frozenset(str(g) for g in gens))
                if key in seen_configs:
                    continue
                seen_configs.add(key)

                entry = cayley_graph_to_dict(G, gens, gid, order, is_ab)
                dataset.append(entry)
                total_graphs += 1

            print("  {} | abelian={} | {} graph(s) added".format(
                gid, is_ab, len(gen_sets)))

    print("-" * 50)
    print("Total groups processed : {}".format(total_groups))
    print("Total graphs in dataset: {}".format(total_graphs))
    abelian_count = sum(1 for d in dataset if d["is_abelian"])
    print("  Abelian    : {}".format(abelian_count))
    print("  Non-abelian: {}".format(total_graphs - abelian_count))

    out_path = "cayley_dataset.json"
    with open(out_path, "w") as f:
        json.dump(dataset, f, indent=2)
    print("\nSaved to: {}".format(out_path))


main()
