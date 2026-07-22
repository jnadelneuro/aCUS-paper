TITLE A-current (Li et al 2011, from Warman 1994)
: LF Ce cells: +20mV shift on inactivation

NEURON {
    SUFFIX ia_li
    USEION k READ ek WRITE ik
    RANGE gbar, hvshift, ik
}

UNITS {
    (mA) = (milliamp)
    (mV) = (millivolt)
    (S)  = (siemens)
}

PARAMETER {
    gbar = 0.0057 (S/cm2)
    hvshift = 0 (mV) : shift already in Table S2 constants
}

ASSIGNED {
    v (mV)
    ek (mV)
    ik (mA/cm2)
    minf hinf
    mtau (ms) htau (ms)
}

STATE { m h }

BREAKPOINT {
    SOLVE states METHOD cnexp
    ik = gbar * m * h * (v - ek)
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
    LOCAL am, bm, ah, bh, vhs
    : activation (no shift)
    if (fabs(v + 20) < 1e-6) {
        am = 0.05 * 15
    } else {
        am = 0.05 * (v + 20) / (1 - exp(-(v + 20) / 15))
    }
    if (fabs(v + 10) < 1e-6) {
        bm = 0.1 * 8
    } else {
        bm = 0.1 * (v + 10) / (exp((v + 10) / 8) - 1)
    }
    minf = am / (am + bm)
    mtau = 1 / (am + bm)
    : inactivation (with hvshift)
    vhs = v - hvshift
    : Table S2: alpha_h = 0.00012 / exp((V-2)/15) = 0.00012 * exp(-(V-2)/15)
    ah = 0.00012 * exp(-(vhs - 2) / 15)
    bh = 0.048 / (1 + exp(-(vhs + 53) / 12))
    hinf = ah / (ah + bh)
    htau = 1 / (ah + bh)
}
