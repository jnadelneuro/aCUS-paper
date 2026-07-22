TITLE Delayed rectifier K (Li et al 2011, from Durstewitz 2000)

NEURON {
    SUFFIX kdr_li
    USEION k READ ek WRITE ik
    RANGE gbar, ik
}

UNITS {
    (mA) = (milliamp)
    (mV) = (millivolt)
    (S)  = (siemens)
}

PARAMETER {
    gbar = 0.01 (S/cm2)
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
    ik = gbar * m*m*m*m * (v - ek)
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
    if (fabs(v - 13) < 1e-6) {
        am = 0.018 * 25
    } else {
        am = 0.018 * (v - 13) / (1 - exp(-(v - 13) / 25))
    }
    if (fabs(v - 23) < 1e-6) {
        bm = 0.0054 * 12
    } else {
        bm = 0.0054 * (v - 23) / (exp((v - 23) / 12) - 1)
    }
    minf = am / (am + bm)
    mtau = 1 / (am + bm)
}
