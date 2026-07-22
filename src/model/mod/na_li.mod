TITLE Na current for Ce neurons (Li et al 2011, from Durstewitz 2000)
: vh=+5mV shift for Ce neurons applied

NEURON {
    SUFFIX na_li
    USEION na READ ena WRITE ina
    RANGE gbar, ina
}

UNITS {
    (mA) = (milliamp)
    (mV) = (millivolt)
    (S)  = (siemens)
}

PARAMETER {
    gbar = 0.12 (S/cm2)
    vh = 5 (mV) : voltage shift for Ce neurons
}

ASSIGNED {
    v (mV)
    ena (mV)
    ina (mA/cm2)
    minf hinf
    mtau (ms) htau (ms)
}

STATE { m h }

BREAKPOINT {
    SOLVE states METHOD cnexp
    ina = gbar * m*m*m * h * (v - ena)
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
    LOCAL am, bm, ah, bh, vs
    vs = v - vh
    : activation
    if (fabs(vs + 28) < 1e-6) {
        am = 0.2816 * 9.3
    } else {
        am = 0.2816 * (vs + 28) / (1 - exp(-(vs + 28) / 9.3))
    }
    if (fabs(vs + 1) < 1e-6) {
        bm = 0.2464 * 6
    } else {
        bm = 0.2464 * (vs + 1) / (exp((vs + 1) / 6) - 1)
    }
    minf = am / (am + bm)
    mtau = 1 / (am + bm)
    : inactivation
    ah = 0.098 * exp(-(vs + 43.1) / 20)
    bh = 1.4 / (1 + exp(-(vs + 13.1) / 10))
    hinf = ah / (ah + bh)
    htau = 1 / (ah + bh)
}
