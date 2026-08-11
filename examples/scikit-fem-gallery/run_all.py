"""Run the initial scikit-fem Gallery compatibility examples."""

from ex01_poisson import compare as compare_ex01
from ex09_poisson_3d import compare as compare_ex09
from ex19_heat_equation import compare as compare_ex19


def main() -> None:
    comparisons=(compare_ex01(),compare_ex09(),compare_ex19())
    for comparison in comparisons:
        comparison.assert_matches()
        print(comparison.summary())


if __name__=="__main__":
    main()
