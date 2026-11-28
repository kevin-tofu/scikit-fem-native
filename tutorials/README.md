# Unfitted and discontinuous finite-element tutorials

These tutorials validate geometry and interface machinery before moving to
more demanding physics:

1. [Wheel level-set optimization and GridapTopOpt comparison](wheel-levelset/README.md)
   records a 30-step comparison and its remaining shape divergence.
2. [Mode I crack with XFEM](mode-i/README.md) validates a vector displacement
   enrichment against the analytic Williams field and a fitted stress-intensity
   factor for a stationary crack.
3. [Load-controlled Mode-I crack growth with XFEM](fracture_mode_i/README.md)
   advances a level-set crack when the fitted stress intensity reaches the
   specified toughness.
4. [Paris-law fatigue crack growth](fracture_mode_i/fatigue/README.md) adds a
   cycle counter to the same quasistatic XFEM solve.
