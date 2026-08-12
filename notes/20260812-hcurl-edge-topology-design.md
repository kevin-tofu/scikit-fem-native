# H(curl) phase 0: oriented edge topology

## Scope

This phase defines topology only.  It does not add an edge-owned finite-element
space, Nédélec basis functions, Piola mappings, curl evaluation, or assembly.
Accordingly, `space.hcurl` and `dof.edge` remain planned.  The narrower
`topology.oriented_edge` capability is experimental.

## Global edge identity and orientation

Every global edge is identified by the ordered pair

```text
(min(global_vertex_a, global_vertex_b),
 max(global_vertex_a, global_vertex_b))
```

The same ascending pair defines its global tangent orientation.  Edge IDs are
assigned deterministically by first traversal of cells and their documented
local edge order.  Algorithms must not attach mathematical meaning to the
numeric edge ID; the vertex pair is the stable identity.

For cell `K` and local edge `e`, the topology stores:

```text
element_edges[e, K]       -> global edge ID
element_edge_signs[e, K]  -> +1 or -1
```

The sign is `+1` when the directed local vertex pair agrees with the ascending
global pair and `-1` when it is reversed.  This is the sign later applied to an
edge-oriented basis/DOF, not an attempt to modify cell connectivity.

## Local edge order

Triangle:

```text
(0, 1), (1, 2), (2, 0)
```

Tetrahedron:

```text
(0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3)
```

The directed `(2, 0)` edge is intentional.  Local order follows an oriented
reference simplex boundary, while global identity always uses ascending
vertex IDs.  Keeping these two concepts separate is the reason signs exist.

## Boundary edges

In two dimensions, boundary edges are boundary facets.  In three dimensions,
boundary edges are the union of all edges belonging to boundary triangular
facets.  `boundary_edge_ids` returns unique sorted global edge IDs.  Later
boundary-region support should retain region ownership rather than only this
global union.

## Invariants fixed by tests

- every global edge contains two distinct ascending vertex IDs;
- every local edge maps to exactly one global edge;
- every local sign is `+1` or `-1`;
- applying the sign to the global pair reconstructs the directed local pair;
- reordering cell vertices preserves the set of global edge identities;
- every edge of a single tetrahedron is a boundary edge;
- the diagonal shared by two triangles is not a boundary edge.

## Next phase: Nédélec reference element

The next implementation must remain independent of physical mesh mapping.  It
should define the lowest-order first-family Nédélec triangle element on the
reference triangle and verify its edge moments and reference curl.  Only after
those tests pass should covariant Piola mapping be introduced.

The required sequence is:

1. reference basis and directed edge moment functionals;
2. orientation sign application to local basis/DOF values;
3. covariant Piola value transformation;
4. physical curl transformation;
5. H(curl) mass matrix comparison;
6. curl-curl matrix comparison;
7. boundary edge DOF selection.

Do not reuse nodal `ElementVector` as an H(curl) element.  Matching array
dimensions would not provide tangential continuity or the correct mapping.
