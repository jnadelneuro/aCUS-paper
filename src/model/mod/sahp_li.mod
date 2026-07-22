TITLE Slow AHP current (Li et al 2011, from Warman 1994)
: Exact alpha/beta from Table S2
: [Ca] in formula is in uM; NEURON cai is in mM, so multiply by 1000

NEURON {
    SUFFIX sahp_li
    USEION k READ ek WRITE ik
    USEION ca READ cai
    RANGE gbar, ik
}

UNITS {
    (mA) = (milliamp)
    (mV) = (millivolt)
    (S)  = (siemens)
    (mM) = (milli/liter)
}

PARAMETER {
    gbar = 0.0015 (S/cm2)
}

ASSIGNED {
    v (mV)
    ek (mV)
    cai (mM)
    ik (mA/cm2)
    minf
}

STATE { m }

BREAKPOINT {
    SOLVE states METHOD cnexp
    ik = gbar * m * (v - ek)
}

INITIAL {
    rates()
    m = minf
}

DERIVATIVE states {
    rates()
    m' = (minf - m) / 48
}

PROCEDURE rates() {
    LOCAL am, bm, lca, ca_um
    : Convert mM to uM for formula
    ca_um = cai * 1000
    if (ca_um < 1e-6) {
        lca = -6
    } else {
        lca = log(ca_um) / log(10)
    }
    am = 0.0048 / exp(-5 * lca + 17.5)
    bm = 0.012 / exp(2 * lca + 20)
    if ((am + bm) < 1e-30) {
        minf = 0
    } else {
        minf = am / (am + bm)
    }
}
