TITLE H-current (Li et al 2011, from Womble & Moises 1993)

NEURON {
    SUFFIX ih_li
    NONSPECIFIC_CURRENT ih
    RANGE gbar, eh
}

UNITS {
    (mA) = (milliamp)
    (mV) = (millivolt)
    (S)  = (siemens)
}

PARAMETER {
    gbar = 0.00003 (S/cm2)
    eh = -43 (mV)
}

ASSIGNED {
    v (mV)
    ih (mA/cm2)
    minf
    mtau (ms)
}

STATE { m }

BREAKPOINT {
    SOLVE states METHOD cnexp
    ih = gbar * m * (v - eh)
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
    minf = 1 / (1 + exp((v + 89.2) / 9.5))
    mtau = 1727 * exp(0.019 * v)
}
