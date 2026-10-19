# CayleyGraphSage.py
# Use inside SageMath (Jupyter Sage kernel or `sage -python`) so group types have
# `.gap()`, `.cayley_graph()`, etc.


def CayleyGraphSage(G, generators=None):
    """
    Return the (right) Cayley graph of the finite group ``G`` as a Sage graph.

    Edges follow Sage's ``cayley_graph``: x -> x * g for each generator g in the
    chosen generating set (and expanded inverses as Sage does internally).

    If ``generators`` is None, try GAP's minimal generating set, else ``G.gens()``.
    """
    if generators is None:
        try:
            gap_gens = list(G.gap().MinimalGeneratingSet())
            generators = [G(g) for g in gap_gens]
        except Exception:
            generators = list(G.gens())
    # ``simple=True`` collapses multi-edges when the same directed edge appears twice
    return G.cayley_graph(generators=generators, simple=True)
