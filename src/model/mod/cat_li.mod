TITLE LVA T-type Ca current (Li et al 2011, from Inoue & Strowbridge 2008)
: Shifts for Ce LTB: -15mV activation, -10mV inactivation

NEURON {
    SUFFIX cat_li
    USEION ca READ eca WRITE ica
    RANGE gbar, mvshift, hvshift, ica
}

UNITS {
    (mA) = (milliamp)
    (mV) = (millivolt)
    (S)  = (siemens)
}

PARAMETER {
    gbar = 0.0001 (S/cm2)
    mvshift = -15 (mV) : activation shift
    hvshift = -10 (mV) : inactivation shift
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
    LOCAL vm, vh
    vm = v - mvshift : shift activation
    vh = v - hvshift : shift inactivation
    minf = 1 / (1 + exp(-(vm + 59) / 5.5))
    mtau = 3.5 / (exp(-(vm + 45) / 15) + exp((vm + 45) / 15)) + 1.5
    hinf = 1 / (1 + exp((vh + 80) / 4))
    htau = 40 / (exp(-(vh + 60) / 15) + exp((vh + 60) / 15)) + 10
}
