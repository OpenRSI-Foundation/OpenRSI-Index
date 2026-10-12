# Initial research notebook: incomplete

The target is the zero-field, simple-cubic, uniform nearest-neighbor ferromagnetic
Ising model, with spins ±1, each undirected bond counted once, K=J/(k_B T), and
p(K)=lim log Z/N for periodic volumes. The goal includes all finite K>=0 and the
leading singular laws on both sides of the transition.

The supplied model.py contains only p(K)=log(2)+3 K^2/2. It illustrates the local
zero-coupling anchors, not a controlled global pressure construction. No global
remainder bound, thermodynamic-limit justification, independent transition
location, or critical singular law is supplied. There is no claim that the
starter passes any scientific rubric.

Replace or extend this notebook with the actual construction, derivations,
assumptions and dependencies, accuracy arguments, independent consequences,
critical analysis, reproducibility instructions, and explicit remaining gaps.
Use any appropriate method. The optional pressure(K) function is a convenience
for public finite probes, not a restriction on the space of scientific theories.
