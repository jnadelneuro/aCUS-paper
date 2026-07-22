TITLE M-current (Li et al 2011, from Warman 1994)

NEURON {
    SUFFIX im_li
    USEION k READ ek WRITE ik
    RANGE gbar, ik
}

UNITS {
    (mA) = (milliamp)
    (mV) = (millivolt)
    (S)  = (siemens)
}

PARAMETER {
    gbar = 0.00015 (S/cm2)
}

ASSIGNED {
    v (mV)
    ek (mV)
    ik (mA/cm2)
    minf
    mtau (ms)
}

STATE { m }

BREAKPOINT {
    SOLVE states METHOD cnexp
    ik = gbar * m*m * (v - ek)
}

INITIAL {
    rates(v)
    m = minf
}

DERIVATIVE states {
    rates(v)
    m' = (minf - m) / mtau
}

PROCEDURE rates(v (mV)) {
    LOCAL am, bm
    am = 0.016 * exp((v + 52.7) / 23)
    bm = 0.016 * exp(-(v + 52.7) / 18.8)
    minf = am / (am + bm)
    mtau = 1 / (am + bm)
}
