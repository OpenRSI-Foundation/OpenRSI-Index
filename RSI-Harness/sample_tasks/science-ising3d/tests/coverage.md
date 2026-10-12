# Coverage

## Scope

The goal is zero-field equilibrium thermodynamics of the uniform nearest-neighbor ferromagnet on the simple-cubic lattice in the periodic-volume thermodynamic limit. The requested outputs are pressure, energy, heat capacity, the critical coupling and leading singularities on both sides, with high/low-temperature asymptotics and controlled accuracy. Resolving this goal would be a major thermodynamic result. It does not establish the nonzero-field equation of state, magnetization response, all correlation functions, a conformal spectrum, other lattices, disorder or nonequilibrium dynamics. “All 3D Ising is solved” would overstate this scope.

Operationalizations chosen for a well-posed target, recorded in the contract's domain specification: periodic cubes for the finite volumes, each undirected bond counted once, K=J/(k_B T), p(K)=lim log Z/N, energy/J=-p'(K) and heat capacity/k_B=K^2 p''(K), and t=Kc/K-1 for the critical sides. No regime, observable or implication of the target is excluded.

## Claims to rubrics

| Claim | Rubrics | Why these establish the claim |
| --- | --- | --- |
| C01 | R01 | R01 states this claim's result, domain, evidence and pass, fail and unresolved conditions on its own |
| C02 | R02 | R02 states this claim's result, domain, evidence and pass, fail and unresolved conditions on its own |
| C03 | R03 | R03 states this claim's result, domain, evidence and pass, fail and unresolved conditions on its own |
| C04 | R04 | R04 states this claim's result, domain, evidence and pass, fail and unresolved conditions on its own |
| C05 | R05 | R05 states this claim's result, domain, evidence and pass, fail and unresolved conditions on its own |
| C06 | R06 | R06 states this claim's result, domain, evidence and pass, fail and unresolved conditions on its own |
| C07 | R07 | R07 states this claim's result, domain, evidence and pass, fail and unresolved conditions on its own |
| C08 | R08 | R08 states this claim's result, domain, evidence and pass, fail and unresolved conditions on its own |
| C09 | R09 | R09 states this claim's result, domain, evidence and pass, fail and unresolved conditions on its own |
| C10 | R10 | R10 states this claim's result, domain, evidence and pass, fail and unresolved conditions on its own |
| C11 | R11 | R11 states this claim's result, domain, evidence and pass, fail and unresolved conditions on its own |
| C12 | R12 | R12 states this claim's result, domain, evidence and pass, fail and unresolved conditions on its own |
| C13 | R13 | R13 states this claim's result, domain, evidence and pass, fail and unresolved conditions on its own |
| C14 | R14 | R14 states this claim's result, domain, evidence and pass, fail and unresolved conditions on its own |
| C15 | R15 | R15 states this claim's result, domain, evidence and pass, fail and unresolved conditions on its own |
| C16 | R16 | R16 states this claim's result, domain, evidence and pass, fail and unresolved conditions on its own |
| C17 | R17 | R17 states this claim's result, domain, evidence and pass, fail and unresolved conditions on its own |
| C18 | R18 | R18 states this claim's result, domain, evidence and pass, fail and unresolved conditions on its own |
| C19 | R19 | R19 states this claim's result, domain, evidence and pass, fail and unresolved conditions on its own |
| C20 | R20 | R20 states this claim's result, domain, evidence and pass, fail and unresolved conditions on its own |
| C21 | R21 | R21 states this claim's result, domain, evidence and pass, fail and unresolved conditions on its own |
| C22 | R22 | R22 states this claim's result, domain, evidence and pass, fail and unresolved conditions on its own |
| C23 | R23 | R23 states this claim's result, domain, evidence and pass, fail and unresolved conditions on its own |
| C24 | R24 | R24 states this claim's result, domain, evidence and pass, fail and unresolved conditions on its own |
| C25 | R25 | R25 states this claim's result, domain, evidence and pass, fail and unresolved conditions on its own |
| C26 | R26 | R26 states this claim's result, domain, evidence and pass, fail and unresolved conditions on its own |

| Required scientific result | Rubrics |
| --- | --- |
| Actual finite-lattice model correspondence | R01 |
| Essential model-to-construction derivation and applicable closure | R02 |
| Candidate construction reaches the model thermodynamic limit | R03 |
| Pressure over the ordinary high/low phases separately | R04, R05 |
| Energy over the ordinary high/low phases separately | R06, R07 |
| Heat capacity over the ordinary high/low phases separately | R08, R09 |
| Pressure high/low-temperature endpoint laws separately | R10, R11 |
| Energy high/low-temperature endpoint laws separately | R12, R13 |
| Heat-capacity high/low-temperature endpoint laws separately | R14, R15 |
| Kc obtained from this construction | R16 |
| Critical pressure value and appropriate regular background | R17 |
| Critical energy value and matching one-sided limits | R18 |
| Leading singular pressure law on each critical side | R19, R20 |
| Leading singular energy law on each critical side | R21, R22 |
| Leading singular heat-capacity law on each critical side | R23, R24 |
| One construction across branches, parameters and scientific outputs | R25 |
| Independent reproduction/rederivation of decisive science | R26 |

## Why all 26 cover the target

Every requested conclusion is required: ordinary-phase criteria cover every compact subset of their open phases, including the intermediate range far from either endpoint; endpoint criteria cover K approaching zero and infinity; critical value and side-law criteria cover Kc and its neighborhoods. Together these cover all finite K>=0 and the requested asymptotics. Heat capacity may diverge at Kc; no fabricated finite critical value is required.

Every rubric is a substantive part of the target: each is a separately falsifiable scientific result, with no documentation, setup or file-presence points. An earlier six-criterion design combined separately falsifiable claims (a candidate could reproduce endpoint coefficients while lacking ordinary-phase coverage or a critical derivation); this granularity separates them. The count 26 follows the mapping above, not a quota. The rubrics are separately adjudicable, not logically independent: reused valid reasoning is allowed, and an invalid essential closure can undermine several dependent claims, which the reviewer must trace. Each derivative criterion needs its own applicability, error or remainder control and corroboration, rather than passing because pressure was accepted. If the candidate supplies exact applicable mathematics, unrelated numerical refinement experiments are unnecessary.

Kc, critical exponents and amplitudes must emerge from the same supported construction. A literature value may be used as counterevidence or corroboration, but cannot both supply an input and independently validate its claimed derivation. A fitted curve with an imposed critical exponent does not establish that exponent. Regular backgrounds, correction terms, limit ordering and uncertainties must distinguish the leading singular term from plausible alternatives. Numerical universality evidence is not an exact identity for lattice-specific amplitudes. No full correction spectrum or nonzero-field physics is silently added.

Estimation standard: no numerical precision is imposed. A whole-phase or asymptotic claim passes on an estimate computed from the candidate's own construction with honestly derived uncertainty that resolves the observable or leading term; loose enclosures and growing error bars cannot establish it. A rigorously stated or scientifically supported approximation can qualify when its scope, refinement and asymptotic errors genuinely support the result.

## Implemented checks versus scientific verification

The Judge recomputes two optional routes offline: `probe.py` (the candidate's own bounded reproduction) and finite pressure and derivative probes of `model.py:pressure(K)` at K in {0, 0.01, 0.1, 0.2, 0.3, 1, 8}. They establish only that the candidate's code produces the recorded values in the clean image; no domain checker issues a verdict. Every rubric verdict is model review by the run's own Agent, after a clear integrity audit, under the public `reviewer.md`. Automatic review can misjudge subtle mathematics, evidence independence and reference applicability; all 26 passes mean completion under this public evidence protocol, not a formal guarantee of mathematical truth. The integrity audit is a precondition of any count, not a 27th point, and all 26 passes must concern the same frozen candidate.

## Launch blockers

none

## Assets and sources

[Friedli and Velenik, Chapter 3, Theorem 3.6](https://www.unige.ch/math/folks/velenik/smbook/Ising_Model.pdf) defines the finite spin model, periodic volumes, pressure limit and convexity. Existence of the model pressure is distinct from establishing a submitted approximation or its derivatives. This supports separating model, construction, limit and observable claims.

[Their Chapter 5, sections 5.7.3–5.7.4](https://www.unige.ch/math/folks/velenik/smbook/Cluster_Expansion.pdf) gives high-temperature graph and low-temperature contour representations with applicable convergence domains. Endpoint expansions therefore cannot silently become full-phase or critical solutions. The anchors below follow from these representations and the task normalization; they are necessary checks inside substantive claims, not separate scored points:

- High temperature: p=log(2)+3K^2/2+11K^4/4+O(K^6). There are three bonds and three elementary plaquettes per site; 3 log(cosh K) supplies -K^4/4 and the plaquettes supply 3K^4. Derivative expansions give e=-3K-11K^3+O(K^5), c=3K^2+33K^4+O(K^6), provided differentiation/remainders are justified.
- Low temperature: one flipped spin breaks six bonds, costing 12J; the leading pressure correction is exp(-12K). The next adjacent-pair excitation costs 20J. The supported low-temperature expansion yields p=3K+exp(-12K)+O(exp(-20K)); with justified derivatives, e=-3+12 exp(-12K)+O(exp(-20K)) and c=144K^2 exp(-12K)+O(K^2 exp(-20K)). Finite-volume ground-state degeneracy is subextensive after the thermodynamic limit.

[Hasenbusch and Pinn, direct cubic-Ising study, arXiv:cond-mat/9706003v1, sections 1 and 2.2](https://arxiv.org/html/cond-mat/9706003) explicitly separates regular backgrounds and side-dependent amplitudes, and relates singular free energy, energy and specific heat by differentiation. Its fitted estimates are scientific reference evidence with normalization, finite-size and systematic limitations, not exact target answers. In this task t=Kc/K-1 equals (T-Tc)/Tc: for p_s=B_±|t|^(2-alpha), derivative laws have energy exponent 1-alpha and heat-capacity exponent -alpha. Applying c=K^2 p'' fixes the leading coefficient to B_±(2-alpha)(1-alpha) because K^2(dt/dK)^2 tends to one at Kc. This relation alone does not establish the candidate's exponent, lattice amplitudes or differentiated remainder.
