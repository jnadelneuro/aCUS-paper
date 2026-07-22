TITLE Ca dynamics (Li et al 2011, from Warman 1994)

NEURON {
    SUFFIX cadyn_li
    USEION ca READ ica WRITE cai
    RANGE f, tau_ca, ca_rest, shell
}

UNITS {
    (mA)    = (milliamp)
    (mM)    = (milli/liter)
    (um)    = (micrometer)
    FARADAY = (faraday) (coulombs)
}

PARAMETER {
    f = 0.024
    shell = 1   (um)
    tau_ca = 80 (ms)
    ca_rest = 5e-5 (mM)
}

ASSIGNED {
    ica (mA/cm2)
}

STATE { cai (mM) }

BREAKPOINT {
    SOLVE states METHOD cnexp
}

INITIAL {
    cai = ca_rest
}

DERIVATIVE states {
    cai' = -f * ica / (2 * FARADAY * shell * 1e-4) - (cai - ca_rest) / tau_ca
}
