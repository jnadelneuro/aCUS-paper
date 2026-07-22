TITLE HVA L-type Ca current (Li et al 2011, from Durstewitz 2000)

NEURON {
    SUFFIX cal_li
    USEION ca READ eca WRITE ica
    RANGE gbar, ica
}

UNITS {
    (mA) = (milliamp)
    (mV) = (millivolt)
    (S)  = (siemens)
}

PARAMETER {
    gbar = 0.0001 (S/cm2)
}

ASSIGNED {
    v (mV)
    eca (mV)
    ica (mA/cm2)
    minf hinf
    mtau (ms) htau (ms)
}

STATE { m h }

BREAKPOINT {
    SOLVE states METHOD cnexp
    ica = gbar * m*m * h * (v - eca)
}

INITIAL {
    rates(v)
    m = minf
    h = hinf
}

DERIVATIVE states {
    rates(v)
    m' = (minf - m) / mtau
    h' = (hinf - h) / htau
}

PROCEDURE rates(v (mV)) {
    LOCAL x
    minf = 1 / (1 + exp(-(v + 24.6) / 11.3))
    : tau_m = 1.25 * sech(-0.031*(v+37.1)) = 1.25 / cosh(0.031*(v+37.1))
    x = 0.031 * (v + 37.1)
    mtau = 1.25 / cosh(x)
    if (mtau < 0.01) { mtau = 0.01 }
    hinf = 1 / (1 + exp((v + 12.6) / 18.9))
    htau = 420
}
