import pytest

import skfemntv


def test_capability_registry_declares_current_and_future_space_boundaries():
    assert skfemntv.supports("space.h1")
    assert not skfemntv.supports("space.hcurl")
    assert skfemntv.get_capability("space.hcurl").status is skfemntv.CapabilityStatus.PLANNED
    assert "edge DOFs" in skfemntv.get_capability("space.hcurl").detail


def test_experimental_capability_requires_explicit_opt_in():
    assert not skfemntv.supports("space.l2_dg")
    assert skfemntv.supports("space.l2_dg", include_experimental=True)
    assert skfemntv.require_capability(
        "space.l2_dg", include_experimental=True
    ).name == "space.l2_dg"


def test_require_capability_reports_status_and_reason():
    with pytest.raises(
        skfemntv.UnsupportedCapabilityError,
        match=r"space\.hcurl.*planned.*oriented edge DOFs",
    ):
        skfemntv.require_capability("space.hcurl")


def test_capability_listing_is_stable_and_filterable():
    names = [item.name for item in skfemntv.capabilities()]
    assert names == sorted(names)
    assert all(item.category == "mapping" for item in skfemntv.capabilities(category="mapping"))
    assert len(skfemntv.CAPABILITY_REGISTRY) == len(names)


def test_unknown_capability_is_not_treated_as_unsupported_known_feature():
    with pytest.raises(KeyError, match="unknown skfemntv capability"):
        skfemntv.supports("space.not-real")
