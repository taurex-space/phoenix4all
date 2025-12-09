import pytest


def test_taurex3_detect():
    """Tests taurex3 to see if it runs."""
    pytest.importorskip("taurex")

    from taurex.parameter.classfactory import ClassFactory

    from phoenix4all.taurex import Phoenix4AllStar

    cf = ClassFactory()

    assert Phoenix4AllStar in cf.starKlasses
