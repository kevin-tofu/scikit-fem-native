from dataclasses import dataclass


@dataclass(frozen=True)
class LinearElasticity:
    kernel_name="linear_elastic"
    state_size=0
    state_fields=()
    young_modulus: float
    poisson_ratio: float

    def __post_init__(self) -> None:
        if self.young_modulus <= 0:
            raise ValueError("young_modulus must be positive")
        if not -1.0 < self.poisson_ratio < 0.5:
            raise ValueError("poisson_ratio must be in (-1, 0.5)")


@dataclass(frozen=True)
class NeoHookean:
    mu: float
    lmbda: float

    def __post_init__(self) -> None:
        if self.mu <= 0:
            raise ValueError("mu must be positive")
        if self.lmbda < 0:
            raise ValueError("lmbda must be non-negative")

    @classmethod
    def from_young_poisson(
        cls, young_modulus: float, poisson_ratio: float
    ) -> "NeoHookean":
        if young_modulus <= 0:
            raise ValueError("young_modulus must be positive")
        if not -1.0 < poisson_ratio < 0.5:
            raise ValueError("poisson_ratio must be in (-1, 0.5)")
        return cls(
            mu=young_modulus / (2.0 * (1.0 + poisson_ratio)),
            lmbda=young_modulus
            * poisson_ratio
            / ((1.0 + poisson_ratio) * (1.0 - 2.0 * poisson_ratio)),
        )


# Backward-compatible topology-specific spellings.  Kernels themselves are
# topology-independent; the Basis selects Tet4 or Hex8 integration.
LinearElasticTet4 = LinearElasticity
LinearElasticHex8 = LinearElasticity
NeoHookeanTet4 = NeoHookean
NeoHookeanHex8 = NeoHookean
