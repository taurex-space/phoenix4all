import pytest


def test_taurex3_detect():
    """Tests taurex3 to see if it runs."""
    pytest.importorskip("taurex")

    from taurex.parameter.classfactory import ClassFactory

    from phoenix4all.taurex import Phoenix4AllStar

    cf = ClassFactory()

    assert Phoenix4AllStar in cf.starKlasses


def test_keyword():
    """Tests that the keyword is correctly registered."""
    pytest.importorskip("taurex")

    from taurex.parameter.classfactory import ClassFactory

    from phoenix4all.taurex import Phoenix4AllStar

    cf = ClassFactory()

    star = cf.find_klass_from_keyword("phoenix4all")

    assert Phoenix4AllStar == star


def _fake_get_spectrum(calls):
    """A get_spectrum stand-in that records the requested temperature."""
    import numpy as np
    from astropy import units as u

    def fake(*, teff, logg, feh, alpha, **kwargs):
        calls.append(teff)
        wl = np.linspace(0.3, 20.0, 500) * u.um
        return wl, np.full(500, float(teff)) * u.W / u.m**2 / u.um

    return fake


def test_initialize_reuses_sed_until_grid_changes(monkeypatch):
    """Same parameters and grid reuse the last SED; a different grid reloads."""
    pytest.importorskip("taurex")
    import numpy as np

    import phoenix4all.taurex as p4a_taurex

    calls = []
    monkeypatch.setattr(p4a_taurex, "get_spectrum", _fake_get_spectrum(calls))
    star = p4a_taurex.Phoenix4AllStar(temperature=5800)
    wngrid = np.linspace(600, 20000, 300)

    star.initialize(wngrid)
    first = star.sed.copy()
    # In-place changes to star.sed, after a computed or a cached call, must not reach the cache.
    star.sed *= 2.0
    star.initialize(wngrid.copy())  # equal grid, different array
    np.testing.assert_array_equal(star.sed, first)
    star.sed *= 2.0
    star.initialize(wngrid)
    np.testing.assert_array_equal(star.sed, first)
    assert calls == [5800]

    star.initialize(np.linspace(1000, 10000, 50))
    assert calls == [5800, 5800]


def test_initialize_reloads_when_temperature_changes(monkeypatch):
    """A new temperature (e.g. when fitting it) reloads the spectrum."""
    pytest.importorskip("taurex")
    import numpy as np

    import phoenix4all.taurex as p4a_taurex

    calls = []
    monkeypatch.setattr(p4a_taurex, "get_spectrum", _fake_get_spectrum(calls))
    star = p4a_taurex.Phoenix4AllStar(temperature=5800)
    wngrid = np.linspace(600, 20000, 300)

    star.initialize(wngrid)
    star.temperature = 6000
    star.initialize(wngrid)
    assert calls == [5800, 6000]
