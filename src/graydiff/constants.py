"""Shared physical and grid constants for the Gray-Scott system.

F/K ranges and the two named checkpoints come directly from the project spec
(`Inverse_Design_Differentiable_Surrogate_Spec.docx`, Phase 0 / Phase 1).
"""

# Diffusion rates (U spreads faster than V — this asymmetry is what lets
# V's autocatalysis outrun U's replenishment and create structure).
DU = 0.16
DV = 0.08

# The region of (F, k) space the whole project operates in. Chosen to span
# the interesting Turing-pattern regime (spots/stripes/mazes/death), per spec.
F_RANGE = (0.020, 0.070)
K_RANGE = (0.050, 0.070)

# Named checkpoints from the spec — used to validate the solver is correct
# before anything is built on top of it.
SPOTS_CHECKPOINT = {"F": 0.035, "k": 0.065}
MAZES_CHECKPOINT = {"F": 0.029, "k": 0.057}

DEFAULT_GRID_SIZE = 64
