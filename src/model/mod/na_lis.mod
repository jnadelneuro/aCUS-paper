TITLE Na current with slow inactivation for Ce LTB neurons
: Based on Li et al 2011 (from Durstewitz 2000), vh=+5mV shift
: Slow inactivation gate (s) from Qian et al 2014, J Neurophysiol 112:2779-2790
:   Used only for LTB (cluster 2) neurons
:   s_half = -54.8 mV, s_slope = -1.57 mV
:   tau_s = 20 + 160/(1 + exp((v+47.2)/1))
:   Experimental basis: Ding et al 2011, Fernandez & White 2010, Migliore et al 1999

NEURON {
    SUFFIX na_lis
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
    minf hinf sinf
    mtau (ms) htau (ms) stau (ms)
}

STATE { m h s }

BREAKPOINT {
    SOLVE states METHOD cnexp
    ina = gbar * m*m*m * h * s * (v - ena)
}

INITIAL {
    rates(v)
    m = minf
    h = hinf
    s = sinf
}

DERIVATIVE states {
    rates(v)
    m' = (minf - m) / mtau
    h' = (hinf - h) / htau
    s' = (sinf - s) / stau
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
    : fast inactivation
    ah = 0.098 * exp(-(vs + 43.1) / 20)
    bh = 1.4 / (1 + exp(-(vs + 13.1) / 10))
    hinf = ah / (ah + bh)
    htau = 1 / (ah + bh)
    : slow inactivation (Qian et al 2014, Table 1)
    sinf = 1 / (1 + exp((v + 54.8) / 1.57))
    stau = 20 + 160 / (1 + exp((v + 47.2) / 1))
}
