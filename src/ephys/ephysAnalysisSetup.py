import scipy.signal
import numpy as np
import pandas as pd
from ipfx.feature_extractor import SpikeFeatureExtractor, SpikeTrainFeatureExtractor
import matplotlib.pyplot as plt
import plotly.express as px
from scipy.optimize import curve_fit

#! idk if I'm gonna use this class


# class Experiment:
#     def __init__(self, name, protocol, params):
#         self.name = name
#         self.protocol = protocol
#         self.params = params 
#         self.cells = []


def censored_rheobase_value(rheoCurrentDict):
    """Rheobase to assign a cell that fired on no sweep: highest current tested
    + one step. Step = median spacing of the current ladder (fallback 5 pA).
    Returns None if the ladder is empty/unusable (cell then stays uncensored)."""
    currents = sorted(c for c in rheoCurrentDict.values() if c is not None)
    if not currents:
        return None
    diffs = [b - a for a, b in zip(currents[:-1], currents[1:]) if b - a > 0]
    step = float(np.median(diffs)) if diffs else 5.0
    return currents[-1] + step


# %%
class EphysMouse:
    """
     _summary_

    _extended_summary_
    """

    def __init__(self, name, stressCon, sex, proj, use=True, age=None, drugType=None):
        """
        __init__ _summary_

        _extended_summary_

        Args:
            name (_type_): _description_
            stressCon (_type_): _description_
            sex (_type_): _description_
            age (_type_): _description_
            holdingSweeps (list, optional): _description_. Defaults to [].
            use (bool, optional): _description_. Defaults to True.
            cells (list, optional): _description_. Defaults to [].
        """
        self.name = name
        self.stressCon = stressCon
        self.sex = sex
        self.proj = proj
        self._use = use
        self._age = age
        self.cells = []
        self.drugType = drugType
        self._drugSweeps = []

    @property
    def bothSweeps(self):
        # * to add the cells that should be included in holding and in non-holding since they're at -60 at rest so didn't need to repeat
        return self._bothSweeps

    @bothSweeps.setter
    def bothSweeps(self, newList):
        if type(newList) is list:
            self._bothSweeps = newList
        else:
            print('please enter a list of ranges')

    @property
    def holdingSweeps(self):
        return self._holdingSweeps

    @holdingSweeps.setter
    def holdingSweeps(self, newList):
        if type(newList) is list:
            self._holdingSweeps = newList
        else:
            print('please enter a list of ranges')

    @property
    def use(self):
        return self._use

    @use.setter
    def use(self, newUse):
        if type(newUse) is bool:
            self._use = newUse
        else:
            print('please enter a bool value')

    @property
    def age(self):
        # Only a subset of mice (e.g. the timecourse dataset) have an age.
        # getattr keeps this safe for pickles created before _age existed.
        return getattr(self, '_age', None)

    @age.setter
    def age(self, newAge):
        self._age = newAge

    @property
    def drugSweeps(self):
        return self._drugSweeps

    @drugSweeps.setter
    def drugSweeps(self, newList):
        if type(newList) is list:
            self._drugSweeps = newList
        else:
            print('please enter a list of ranges')

    def addCell(self, newCell):
        """
        addCell _summary_

        _extended_summary_

        Args:
            newCell (object of class EphysCell): _description_
        """
        # * newCell should be an instance of the EphysCell class, do this at the end of making each cell
        self.cells.append(newCell)
        # return self #* bring back to chain
# %%


class EphysCell:
    def __init__(self, name, use=True):
        # sweeps should be an instance of the EphysSweep class
        self.name = name
        # self.bead = bead
        self.sweeps = {}
        self.use = use

    
    @property
    def use(self):
        return self._use

    @use.setter
    def use(self, newUse):
        if type(newUse) is bool:
            self._use = newUse
        else:
            print('please enter a bool value')
    
    # def calculateMeanValues(self, data):
    #     # param should be an attribute of the class
    #     # this takes a dict of sweeps then values (e.g. resting membrane potential, input resistance, access resistance) and caclulate's the cell's mean value
    #     meanValue = np.mean(list(data.values()))


    def createCurrentDict(self, data, params, CD_key):
        # this takes a dictionary output from wavesurfer.loadDataFile,
        # makes sure it's from an appropriate protcol, then creates a dictionary with current values for each sweep
        num_sweeps = 0
        currentInjection_Dict = {}
        for key, value in data.items():
            if 'sweep' in key:
                num_sweeps += 1
                if 'i' in params.amplitude_value:
                    i = num_sweeps
                    currentInjection_Dict[key] = eval(params.amplitude_value)
        setattr(self, CD_key, currentInjection_Dict)

    def stripSweep(self, sweepName):
        # onlyNumStr = sweepName.split('_')[1]
        # zerosRemoved = onlyNumStr.lstrip('0')
        # sweepNum = int(zerosRemoved)

        # this turns shit from 'sweep_0010' from that string into the integer 10
        sweepNum = int(sweepName.split('_')[1].lstrip('0'))
        return sweepNum
    def addInputResistance(self, data, params):
        if not hasattr(self, 'inputResistance'):
            self.inputResistance = {}
        sweepsList = {key: value for key,
                      value in data.items() if 'sweep' in key}
        for sweep in sweepsList:
            sweepNum = self.stripSweep(sweep)
            trace = np.transpose(np.array(data[sweep]['analogScans']))[0]
            RsBaselineCurrent = trace[500:1001].mean() # mean of values 50ms before voltage change onset at 0.1sec
            RsSteadyStateCurrent = trace[1350:1650].mean() # mean of values during steady state before voltage goes off
            dCurrent = RsSteadyStateCurrent - RsBaselineCurrent # calculate the change in current in response to voltage step
            dVoltage = float(params.amplitude_value) # -5mv voltage step is the usual, but double check if voltage step is changed in recording 
            inputResistance = 1000 * (dVoltage/dCurrent) # R = V/I  ; mV/pA = G ohm; *1000 to get M Ohm 
            self.inputResistance[sweepNum] = inputResistance
        # print(self.inputResistance)
    
    def addSag(self, data, params):
        # Current-clamp voltage-sag analysis (formerly misnamed "Ih"). Computes a
        # sag ratio per hyperpolarizing current step. The real (voltage-clamp) Ih
        # is measured separately by addIh.
        if not hasattr(self, 'sagData'):
            self.sagData = {
                'sag': [],
                'current amp': [],
                'sweep': []
            }

        sweepsList = {key: value for key,
                      value in data.items() if 'sweep' in key}
        i = 0
        for sweep in sweepsList:
            currentAmps = list(self.sagCurrentDict.values())[i]
            i = i + 1
            if currentAmps == 0:    
                continue

            trace = np.transpose(np.array(data[sweep]['analogScans']))[0]
            total_time = len(trace)/params.sampling_rate
            time = np.linspace(0, total_time, len(trace))
            
            # Protocol timing - adjust these based on your actual protocol
            baseline_duration_ms = int(float(params.delay) * 1000)  # ms before step
            step_duration_ms = int(float(params.stimDuration) * 1000)     # ms of hyperpolarizing step
            
            # Convert to indices
            dt = round((time[1] - time[0]) * 1000, 4)  # ms  # ms per sample
            baseline_samples = int(baseline_duration_ms / dt)
            step_samples = int(step_duration_ms / dt)
            
            # Define key time points
            baseline_end = baseline_samples
            step_start = baseline_end
            step_end = step_start + step_samples
            
            # Ensure we have enough data
            # if step_end >= len(trace):
            #     return None
                
            # Calculate measurements
            baseline_voltage = np.mean(trace[:baseline_end])
            
            # Find peak hyperpolarization during step
            step_trace = trace[step_start:step_end]
            peak_idx_in_step = np.argmin(step_trace)
            peak_hyperpol = step_trace[peak_idx_in_step]
            
            # Steady-state voltage (last 10% of step)
            ss_start = step_end - int(0.1 * step_samples)
            steady_state_voltage = np.mean(trace[ss_start:step_end])
            
            # Sag measurements
            sag_amplitude = peak_hyperpol - steady_state_voltage
            total_deflection = peak_hyperpol - baseline_voltage
            sag_ratio = abs(sag_amplitude) / abs(total_deflection) if abs(total_deflection) > 0 else 0
            
            
            self.sagData['sag'].append(sag_ratio)
            self.sagData['current amp'].append(currentAmps)
            self.sagData['sweep'].append(sweep)


    def addIh(self, data, params):
        """
        Voltage-clamp Ih: the slow inward current that develops during a
        hyperpolarizing voltage step (protocol PA_Ch1_WC_VC_Ih test.cfg).

        The cell holds at params.holding_mV (-70 mV); the analyzed step is a
        hyperpolarizing step beginning at params.step_onset_s for
        params.step_duration_s (here 2.0 s -> 3.5 s, to ~-120 mV). Ih appears
        as a slowly-developing inward (negative) relaxation from an
        instantaneous level (just after the capacitive transient) to a
        steady-state plateau (end of the step).

        Two complementary measures are computed and stored side by side
        (per the literature: window subtraction for amplitude, exponential for
        kinetics):
          Method 1 (exponential fit): fit I(t)=A*exp(-t/tau)+C over the step,
            skipping the first params.cap_skip_s (capacitive transient).
              steadyState = C ; instantaneous = A+C (fit at step onset, post-cap)
              Ih = steadyState - instantaneous (= -A; signed, inward<0)
              tau (ms) and fit R^2 also stored. R^2 is informational only and is
              NOT used to exclude sweeps.
          Method 2 (window subtraction): instantaneous = mean over a short
            (~10 ms) window at the end of the capacitive transient; steadyState
            = mean over the last params.ss_window_s of the step; Ih_win = diff.

        Automatic bad-sweep QC (no manual list). A sweep failing EITHER gate has
        ALL its values set to NaN (so the per-cell mean downstream drops it) and
        is logged in self.IhQC:
          1. Unstable baseline/holding: over the initial holding baseline window
             ([baseline_offset_s, +baseline_window_s], within the 0..step_onset
             holding period), reject if |mean current| > hold_max_pA, or the
             half-to-half drift exceeds baseline_drift_tol (pA).
          2. No steady-state plateau: reject if |mean(last ss_window_s) -
             mean(preceding ss_window_s)| > plateau_tol (pA) -- a dying/unstable
             cell keeps drifting and never plateaus.

        Stores {sweepNum: value} dicts (consumed by extract_scalar downstream):
          fit:    self.Ih, self.IhInstantaneous, self.IhSteadyState,
                  self.IhTau, self.IhR2
          window: self.IhWin, self.IhInstWin, self.IhSsWin
          qc:     self.IhQC  ('pass' or failure reason)

        Optional params attributes (defaults in parens):
            sampling_rate          (10000)
            current_channel_index  (0)
            step_onset_s           (2.0)    hyperpol step start, s
            step_duration_s        (1.5)    step length, s
            cap_skip_s             (0.015)  skip capacitive transient before fit
            ss_window_s            (0.10)   steady-state / plateau window, s
            tau_min_s, tau_max_s   (0.005, 2.0)  mono-exp fit bounds, s
            baseline_offset_s      (0.1)    holding baseline start, s
            baseline_window_s      (0.3)    holding baseline width, s
            hold_max_pA            (500.0)  QC: max |holding current|, pA
            baseline_drift_tol     (100.0)  QC: max holding half-to-half drift, pA
            plateau_tol            (50.0)   QC: max end-of-step drift, pA
        """
        for attr in ('Ih', 'IhInstantaneous', 'IhSteadyState', 'IhTau', 'IhR2',
                     'IhWin', 'IhInstWin', 'IhSsWin', 'IhQC'):
            if not hasattr(self, attr):
                setattr(self, attr, {})

        def exp_decay(t, A, tau, C):
            return A * np.exp(-t / tau) + C

        sr = float(getattr(params, 'sampling_rate', 10000))
        dt = 1.0 / sr
        ch_idx     = int(getattr(params, 'current_channel_index', 0))
        onset_s    = float(getattr(params, 'step_onset_s', 2.0))
        dur_s      = float(getattr(params, 'step_duration_s', 1.5))
        cap_skip_s = float(getattr(params, 'cap_skip_s', 0.015))
        ss_win_s   = float(getattr(params, 'ss_window_s', 0.10))
        tau_min_s  = float(getattr(params, 'tau_min_s', 0.005))
        tau_max_s  = float(getattr(params, 'tau_max_s', 2.0))
        base_off_s = float(getattr(params, 'baseline_offset_s', 0.1))
        base_win_s = float(getattr(params, 'baseline_window_s', 0.3))
        hold_max   = float(getattr(params, 'hold_max_pA', 500.0))
        drift_tol  = float(getattr(params, 'baseline_drift_tol', 100.0))
        plateau_tol = float(getattr(params, 'plateau_tol', 50.0))

        onset_idx    = int(round(onset_s * sr))
        step_end_idx = int(round((onset_s + dur_s) * sr))
        cap_skip     = int(round(cap_skip_s * sr))
        fit_start    = onset_idx + cap_skip
        ss_win       = int(round(ss_win_s * sr))
        inst_win     = int(round(0.010 * sr))   # 10 ms window for window-instantaneous
        base_start   = int(round(base_off_s * sr))
        base_end     = int(round((base_off_s + base_win_s) * sr))

        def nan_out(sweepNum, reason):
            for attr in ('Ih', 'IhInstantaneous', 'IhSteadyState', 'IhTau',
                         'IhR2', 'IhWin', 'IhInstWin', 'IhSsWin'):
                getattr(self, attr)[sweepNum] = np.nan
            self.IhQC[sweepNum] = reason
            print(f'[Ih QC] {self.name} {sweepNum}: excluded ({reason})')

        sweepsList = {k: v for k, v in data.items() if 'sweep' in k}
        for sweep in sweepsList:
            sweepNum = self.stripSweep(sweep)
            scans = np.transpose(np.array(data[sweep]['analogScans']))  # channels x samples
            use_idx = ch_idx if ch_idx < scans.shape[0] else scans.shape[0] - 1
            trace = scans[use_idx]

            # Guard trace length
            if len(trace) < step_end_idx:
                nan_out(sweepNum, 'trace too short')
                continue

            # --- QC gate 1: unstable baseline / holding ---
            base_seg = trace[base_start:base_end]
            hold_mean = float(np.mean(base_seg))
            half = len(base_seg) // 2
            base_drift = abs(float(np.mean(base_seg[half:])) - float(np.mean(base_seg[:half]))) \
                if half > 0 else np.inf
            if abs(hold_mean) > hold_max:
                nan_out(sweepNum, f'holding out of range ({hold_mean:.0f} pA)')
                continue
            if base_drift > drift_tol:
                nan_out(sweepNum, f'baseline drift ({base_drift:.0f} pA)')
                continue

            # --- QC gate 2: no steady-state plateau ---
            ss_seg = trace[step_end_idx - ss_win:step_end_idx]
            prev_seg = trace[step_end_idx - 2 * ss_win:step_end_idx - ss_win]
            plateau_drift = abs(float(np.mean(ss_seg)) - float(np.mean(prev_seg)))
            if plateau_drift > plateau_tol:
                nan_out(sweepNum, f'no plateau (end drift {plateau_drift:.0f} pA)')
                continue

            # --- Method 2: window subtraction (always computable once QC passes) ---
            I_inst_win = float(np.mean(trace[fit_start:fit_start + inst_win]))
            I_ss_win = float(np.mean(ss_seg))
            self.IhInstWin[sweepNum] = I_inst_win
            self.IhSsWin[sweepNum] = I_ss_win
            self.IhWin[sweepNum] = I_ss_win - I_inst_win

            # --- Method 1: bounded mono-exponential fit ---
            seg = trace[fit_start:step_end_idx]
            t_axis = np.arange(len(seg)) * dt
            try:
                p0 = (seg[0] - seg[-1], 0.15, seg[-1])
                bounds = ([-np.inf, tau_min_s, -np.inf], [np.inf, tau_max_s, np.inf])
                popt, _ = curve_fit(exp_decay, t_axis, seg, p0=p0, bounds=bounds, maxfev=10000)
                A, tau, C = popt
                I_inst_fit = A + C            # fit value at step onset (post-cap)
                I_ss_fit = C                  # asymptote
                resid = seg - exp_decay(t_axis, *popt)
                ss_res = float(np.sum(resid ** 2))
                ss_tot = float(np.sum((seg - seg.mean()) ** 2))
                r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
                self.IhInstantaneous[sweepNum] = I_inst_fit
                self.IhSteadyState[sweepNum] = I_ss_fit
                self.Ih[sweepNum] = I_ss_fit - I_inst_fit
                self.IhTau[sweepNum] = tau * 1000.0
                self.IhR2[sweepNum] = r2
            except Exception:
                self.IhInstantaneous[sweepNum] = np.nan
                self.IhSteadyState[sweepNum] = np.nan
                self.Ih[sweepNum] = np.nan
                self.IhTau[sweepNum] = np.nan
                self.IhR2[sweepNum] = np.nan

            self.IhQC[sweepNum] = 'pass'


    def addAccess_andCap(self, data, params):
        """
        Robust calculation of access resistance (Rs), capacitance (C), and membrane tau.

        Reads timing windows from ProtocolParams rather than hardcoded sample indices.
        Bounded exponential fit prevents nonsense tau values. Safety guards on
        trace length and divide-by-zero return NaN instead of crashing.

        Required params attributes (with defaults if missing):
            sampling_rate          (default 10000)
            tp_delay_s             time before step onset, sec   (default 0.10)
            tp_duration_s          step duration, sec            (default 0.10)
            baseline_offset_s      gap before step for baseline  (default 0.05)
            baseline_window_s      baseline window width, sec    (default 0.05)
            amplitude_value        voltage step (mV)             (default -5.0)
            current_channel_index  which analog channel is current (default 0)
        """
        if not hasattr(self, 'capacitance'):
            self.capacitance = {}
        if not hasattr(self, 'accessResistance'):
            self.accessResistance = {}
        if not hasattr(self, 'membraneTau'):
            self.membraneTau = {}

        def exp_decay(t, A, tau, C):
            return A * np.exp(-t / tau) + C

        sr = float(getattr(params, 'sampling_rate', 10000))
        dt = 1.0 / sr

        # Timing windows from params (with sensible defaults)
        tp_delay_s    = float(getattr(params, 'tp_delay_s', 0.10))
        tp_duration_s = float(getattr(params, 'tp_duration_s', 0.10))
        base_off_s    = float(getattr(params, 'baseline_offset_s', 0.05))
        base_win_s    = float(getattr(params, 'baseline_window_s', 0.05))
        step_mV       = float(getattr(params, 'amplitude_value', -5.0))
        ch_idx        = int(getattr(params, 'current_channel_index', 0))

        tp_start_idx = int(round(tp_delay_s * sr))
        tp_end_idx   = int(round((tp_delay_s + tp_duration_s) * sr))

        # Baseline window: ends `base_off_s` before step; width `base_win_s`
        base_start = max(tp_start_idx - int(round(base_off_s * sr)) - int(round(base_win_s * sr)), 0)
        base_end   = max(tp_start_idx - int(round(base_off_s * sr)), base_start + 1)

        # Transient window: 20 ms starting at step onset
        sweepsList = {k: v for k, v in data.items() if 'sweep' in k}
        if not sweepsList:
            return
        n_samp_first = np.asarray(next(iter(sweepsList.values()))['analogScans']).shape[0]
        trans_start = tp_start_idx
        trans_end   = min(tp_start_idx + int(round(0.020 * sr)), n_samp_first)

        # Tau fit window: first 10 ms of transient
        fit_end_idx = min(tp_start_idx + int(round(0.010 * sr)), trans_end)

        for sweep in sweepsList:
            sweepNum = self.stripSweep(sweep)
            scans = np.transpose(np.array(data[sweep]['analogScans']))  # channels × samples
            # Bounds-check channel index; fall back to last channel if needed
            use_idx = ch_idx if ch_idx < scans.shape[0] else scans.shape[0] - 1
            trace = scans[use_idx]

            # Guard against too-short traces
            if len(trace) < max(fit_end_idx, base_end, trans_end):
                self.accessResistance[sweepNum] = np.nan
                self.capacitance[sweepNum]      = np.nan
                self.membraneTau[sweepNum]      = np.nan
                continue

            # Baseline current (mean over pre-step window)
            baseline = float(np.mean(trace[base_start:base_end])) if base_end > base_start \
                else float(np.mean(trace[:tp_start_idx]))

            # Transient: peak of capacitive current after step onset
            transient_segment = trace[trans_start:trans_end]
            dCurrent = float(np.min(transient_segment) - baseline)  # negative for -5 mV step

            # Access resistance: R = V/I (mV/pA = GΩ; *1000 → MΩ)
            if abs(dCurrent) > 1e-9:
                self.accessResistance[sweepNum] = 1000.0 * (step_mV / dCurrent)
            else:
                self.accessResistance[sweepNum] = np.nan

            # Capacitance: ∫(I - baseline) dt / V_step, stored in pF.
            # charge ∫I·dt = pA·s = pC; pC/mV = nF; ×1000 → pF.
            capacitive_current = transient_segment - baseline
            charge_pC = float(np.trapz(capacitive_current, dx=dt))
            if abs(step_mV) > 1e-9:
                self.capacitance[sweepNum] = 1000.0 * (charge_pC / step_mV)
            else:
                self.capacitance[sweepNum] = np.nan

            # Membrane tau: bounded mono-exp fit on first 10 ms of transient
            fit_segment = capacitive_current[:(fit_end_idx - trans_start)]
            time_axis   = np.arange(len(fit_segment)) * dt

            tau_ms = np.nan
            if len(fit_segment) > 5 and np.any(np.isfinite(fit_segment)):
                # Bounds: tau between 0.1 ms and 100 ms (physiological range)
                p0 = (fit_segment[0], 0.005, fit_segment[-1])
                bounds = ([-np.inf, 1e-4, -np.inf], [np.inf, 0.1, np.inf])
                try:
                    popt, _ = curve_fit(exp_decay, time_axis, fit_segment,
                                        p0=p0, bounds=bounds, maxfev=5000)
                    tau_ms = popt[1] * 1000.0
                except Exception:
                    tau_ms = np.nan

            self.membraneTau[sweepNum] = tau_ms




    def addFiringRate(self, data, params):
        """
        addFiringRate takes data from sweeps with current injection and calculates firing rates

        _extended_summary_

        Args:
            data (dict): a dictionary created by ws.LoadDataFile in the matlab engine
            params (instance of class ProtocolParams): parameters should be a ProtocolParams instance with appropriate kwargs. For this,
            the required kwargs are threshold, stimDuration, and samplingRate

        Returns:
            adds the attribite "firingRateData" to the instance of the class. firingRateData is a dictionary with lists for current amplitude, firing rate, and sweep number
        """

        if not hasattr(self, 'firingRateData'):
            self.firingRateData = {
                'current amp': [],
                'firing rate': [],
                'sweep': []
            }

        # data should be a wavesurfer output dictionary with header and associated sweeps

        #
        def findFiringRate(trace, threshold, stimDuration, sampling_rate):
            spikes, spikeProps = scipy.signal.find_peaks(
                trace, height=threshold, distance=(0.005*sampling_rate))
            spikeRate = len(spikes)/float(stimDuration)
            return spikeRate

        traces = [np.transpose(np.array(data[sweep]['analogScans']))[
            0] for sweep in self.FRCurrentDict.keys()]
        firingRates = list(map(lambda trace: findFiringRate(
            trace, params.threshold, params.stimDuration, params. sampling_rate), traces))
        currentAmps = list(self.FRCurrentDict.values())
        sweepNums = map(lambda sweep: self.stripSweep(
            sweep), list(self.FRCurrentDict.keys()))

        self.firingRateData['current amp'].extend(currentAmps)
        self.firingRateData['firing rate'].extend(firingRates)
        self.firingRateData['sweep'].extend(sweepNums)

    def addOptoResponse(self, data, params):
        """
        Quantify evoked synaptic responses to optogenetic light pulses.

        Pulse onsets are hardcoded per protocol via params.pulse_onsets_s
        (e.g. (2.0,) for single-pulse, (2.0, 2.101) for PPR with 100ms ISI).

        Per pulse, computes: peak amplitude, time-to-peak, latency (80-20 line
        method), 20-80 rise time, mono-exponential decay tau, charge transfer.
        A responder gate uses 4 × MAD-based sigma of the pre-pulse baseline
        (robust to spontaneous events). For sweeps with >=2 pulses, computes
        peak2/peak1 ratio (NaN if pulse 1 fails the responder gate). When a
        pulse fails the responder gate, latency and rise time are reported
        as NaN (kinetics aren't meaningful for non-responders).

        Stores per-sweep, per-pulse table as cell.optoSweeps (DataFrame) and
        per-pulse-grouped responder-only means as cell.optoGrouped (DataFrame).
        Accumulates across calls (one .h5 file at a time).

        Required params attributes:
            pulse_onsets_s                          tuple/list of pulse onset times
                                                    in seconds, e.g. (2.0,) or
                                                    (2.0, 2.101). REQUIRED.

        Optional params attributes (defaults in parens):
            sampling_rate              (10000)
            current_channel_index      (0)       analog channel with current trace
            test_pulse_end_s           (None)    end of VC test pulse; baseline starts
                                                 test_pulse_recovery_s after this
            test_pulse_recovery_s      (0.2)
            baseline_pre_pulse_gap_s   (0.01)    end baseline this much before pulse 1
            peak_search_window_s       (0.1)     peak-search window per pulse
            decay_fit_window_s         (0.15)    decay fit window after peak
            charge_window_s            (0.2)     charge integration hard cap
            charge_return_sigmas       (2.0)     "back to baseline" = within this many sigmas
            charge_return_duration_s   (0.01)    for this long continuously
            responder_sigma_threshold  (4.0)     |peak| > threshold * sigma_MAD = responder
            decay_tau_min_s            (0.0005)  fit bounds: 0.5 ms
            decay_tau_max_s            (0.2)     fit bounds: 200 ms
            pulse_label                ('intensity', value)   tuple (col_name, value) tagged
                                                              on each row; or None.
                                                              For per-sweep variation use
                                                              sweep_to_label instead.
            sweep_to_label             ('intensity', {15: range(1,4),
                                                      50: range(4,7), ...})
                                                              Per-sweep label assignment.
                                                              First element is the column
                                                              name; second is a dict mapping
                                                              label_value -> ranges of sweep
                                                              numbers. Sweeps not in any
                                                              range get NaN and a warning
                                                              is printed. Takes precedence
                                                              over pulse_label if both set.
        """
        # Defaults
        sr = float(getattr(params, 'sampling_rate', 10000))
        dt = 1.0 / sr
        cur_idx = int(getattr(params, 'current_channel_index', 0))
        tp_end_s = getattr(params, 'test_pulse_end_s', None)
        tp_recov_s = float(getattr(params, 'test_pulse_recovery_s', 0.2))
        base_gap_s = float(getattr(params, 'baseline_pre_pulse_gap_s', 0.01))
        peak_win_s = float(getattr(params, 'peak_search_window_s', 0.1))
        decay_win_s = float(getattr(params, 'decay_fit_window_s', 0.15))
        charge_win_s = float(getattr(params, 'charge_window_s', 0.2))
        chrg_ret_sig = float(getattr(params, 'charge_return_sigmas', 2.0))
        chrg_ret_dur_s = float(getattr(params, 'charge_return_duration_s', 0.01))
        resp_thresh_sig = float(getattr(params, 'responder_sigma_threshold', 4.0))
        tau_min = float(getattr(params, 'decay_tau_min_s', 0.0005))
        tau_max = float(getattr(params, 'decay_tau_max_s', 0.2))
        pulse_label = getattr(params, 'pulse_label', None)
        sweep_to_label_param = getattr(params, 'sweep_to_label', None)

        # Hardcoded pulse onsets (seconds) for this protocol.
        # E.g. (2.0,) for single-pulse, (2.0, 2.101) for PPR with 100ms ISI.
        pulse_onsets_s = getattr(params, 'pulse_onsets_s', None)
        if pulse_onsets_s is None:
            raise ValueError(
                f"addOptoResponse for cell {self.name}: ProtocolParams must define "
                f"pulse_onsets_s (e.g. (2.0,) for single pulse, (2.0, 2.101) for PPR)."
            )

        # Build per-sweep label lookup from ranges-by-value dict.
        # sweep_to_label_param is ('col_name', {value: range_or_list_of_sweeps, ...})
        label_col_name = None
        sweep_to_label_map = None
        if sweep_to_label_param is not None:
            label_col_name, ranges_by_value = sweep_to_label_param
            sweep_to_label_map = {}
            for val, rng in ranges_by_value.items():
                for sw in rng:
                    sweep_to_label_map[int(sw)] = val
        elif pulse_label is not None:
            label_col_name = pulse_label[0]

        def exp_decay(t, A, tau, C):
            return A * np.exp(-t / tau) + C

        sweepsList = {k: v for k, v in data.items() if 'sweep' in k}
        if not sweepsList:
            return

        rows = []
        for sweep_name in sweepsList:
            sweep_num = self.stripSweep(sweep_name)
            scans = np.transpose(np.array(sweepsList[sweep_name]['analogScans']))
            n_ch, n_samp = scans.shape
            cur = scans[min(cur_idx, n_ch - 1)].astype(float)

            # Pulse onsets are hardcoded per protocol, not detected.
            pulse_starts = [int(round(float(t) * sr)) for t in pulse_onsets_s
                            if int(round(float(t) * sr)) < n_samp]
            if not pulse_starts:
                continue

            # ---- pre-pulse baseline window ----
            first_pulse = pulse_starts[0]
            base_end = max(0, first_pulse - int(round(base_gap_s * sr)))
            if tp_end_s is not None:
                base_start = int(round((tp_end_s + tp_recov_s) * sr))
            else:
                base_start = 0
            base_start = min(max(0, base_start), max(0, base_end - 1))
            if base_end - base_start < int(0.02 * sr):  # need at least 20 ms of baseline
                # not enough room: degrade gracefully, use everything before pulse
                base_start = 0
                base_end = max(int(0.02 * sr), first_pulse)
            base_seg = cur[base_start:base_end]
            baseline = float(np.mean(base_seg))
            base_med = float(np.median(base_seg))
            sigma_mad = 1.4826 * float(np.median(np.abs(base_seg - base_med)))
            threshold = resp_thresh_sig * sigma_mad if sigma_mad > 0 else np.nan

            # ---- per-pulse analysis ----
            n_pulses = len(pulse_starts)
            peaks_signed = []  # for PPR calc
            for p_i, p_start in enumerate(pulse_starts):
                # Local baseline: 20 ms ending 5 ms before this pulse
                local_base_end = max(0, p_start - int(round(0.005 * sr)))
                local_base_start = max(0, local_base_end - int(round(0.020 * sr)))
                local_base = float(np.mean(cur[local_base_start:local_base_end])) \
                    if local_base_end > local_base_start else baseline

                # Peak search: clipped to next pulse - 1 ms or peak_win_s, whichever first
                peak_end = p_start + int(round(peak_win_s * sr))
                if p_i + 1 < n_pulses:
                    peak_end = min(peak_end, pulse_starts[p_i + 1] - int(round(0.001 * sr)))
                peak_end = min(peak_end, n_samp)
                if peak_end <= p_start:
                    continue
                seg = cur[p_start:peak_end] - local_base

                peak_offset = int(np.argmin(seg))  # most-negative (inward at -70 = IPSC)
                peak_val = float(seg[peak_offset])  # signed (negative for inward)
                peak_idx = p_start + peak_offset
                peak_time_ms = (peak_offset / sr) * 1000.0
                is_responder = bool(abs(peak_val) > threshold) if not np.isnan(threshold) else False

                # ---- Raw window storage for buildOptoTemplates (pulse 1 only) ----
                if p_i == 0:
                    if not hasattr(self, 'optoRawWindows'):
                        self.optoRawWindows = {}
                    win_start = max(0, p_start - int(round(0.020 * sr)))
                    win_end   = min(p_start + int(round(0.200 * sr)), n_samp)
                    self.optoRawWindows[sweep_num] = (cur[win_start:win_end] - local_base).astype(float)

                # ---- 80-20 line latency (only meaningful for responders) ----
                latency_ms = np.nan
                rise_2080_ms = np.nan
                if is_responder and peak_offset >= 2 and abs(peak_val) > 0:
                    rising = seg[:peak_offset + 1]
                    target_20 = 0.2 * peak_val
                    target_80 = 0.8 * peak_val
                    # For inward currents, peak_val is negative; we want first crossing
                    # of 20% and 80% magnitude. Use signed comparisons cleanly:
                    if peak_val < 0:
                        idx20 = np.where(rising <= target_20)[0]
                        idx80 = np.where(rising <= target_80)[0]
                    else:
                        idx20 = np.where(rising >= target_20)[0]
                        idx80 = np.where(rising >= target_80)[0]
                    if idx20.size > 0 and idx80.size > 0 and idx20[0] < idx80[0]:
                        t20 = idx20[0] / sr
                        t80 = idx80[0] / sr
                        v20 = float(rising[idx20[0]])
                        v80 = float(rising[idx80[0]])
                        rise_2080_ms = (t80 - t20) * 1000.0
                        # Line through (t20, v20) - (t80, v80); intersect y=0
                        # y = v20 + slope*(t - t20); y=0 → t = t20 - v20/slope
                        if t80 != t20:
                            slope = (v80 - v20) / (t80 - t20)
                            if slope != 0 and np.isfinite(slope):
                                t_intercept = t20 - v20 / slope
                                latency_ms = max(0.0, t_intercept * 1000.0)
                    # Fallback: 3σ-MAD crossing if 80-20 didn't give a valid latency
                    if np.isnan(latency_ms) and not np.isnan(threshold):
                        if peak_val < 0:
                            cross = np.where(rising <= -threshold)[0]
                        else:
                            cross = np.where(rising >= threshold)[0]
                        if cross.size > 0:
                            latency_ms = (cross[0] / sr) * 1000.0

                # ---- decay tau (bounded mono-exp) ----
                decay_tau_ms = np.nan
                decay_end = min(peak_idx + int(round(decay_win_s * sr)), n_samp)
                if p_i + 1 < n_pulses:
                    decay_end = min(decay_end, pulse_starts[p_i + 1] - int(round(0.001 * sr)))
                if decay_end - peak_idx > 5:
                    dec_seg = cur[peak_idx:decay_end] - local_base
                    t_axis = np.arange(len(dec_seg)) * dt
                    # Mono-exp: A·exp(-t/τ) + C
                    p0 = (float(dec_seg[0]), 0.02, float(dec_seg[-1]))
                    bounds = ([-np.inf, tau_min, -np.inf], [np.inf, tau_max, np.inf])
                    try:
                        popt, _ = curve_fit(exp_decay, t_axis, dec_seg,
                                            p0=p0, bounds=bounds, maxfev=5000)
                        tau_s = popt[1]
                        # If fit hit a bound, report NaN (unreliable)
                        if tau_min * 1.01 < tau_s < tau_max * 0.99:
                            decay_tau_ms = tau_s * 1000.0
                    except Exception:
                        pass

                # ---- charge transfer (integrate from pulse onset) ----
                charge_pC = np.nan
                charge_end = min(p_start + int(round(charge_win_s * sr)), n_samp)
                if p_i + 1 < n_pulses:
                    charge_end = min(charge_end, pulse_starts[p_i + 1] - int(round(0.001 * sr)))
                if charge_end > p_start and not np.isnan(threshold) and sigma_mad > 0:
                    chrg_seg = cur[p_start:charge_end] - local_base
                    # Find first sustained return-to-baseline (within chrg_ret_sig*sigma
                    # for chrg_ret_dur_s continuous)
                    ret_threshold = chrg_ret_sig * sigma_mad
                    ret_dur_samples = int(round(chrg_ret_dur_s * sr))
                    cutoff = len(chrg_seg)
                    if ret_dur_samples > 0 and len(chrg_seg) >= ret_dur_samples:
                        within = np.abs(chrg_seg) <= ret_threshold
                        # rolling minimum: a window of `ret_dur_samples` is fully within
                        # if cumulative sum of (1 - within) over that window is 0.
                        # Simpler: scan for first index where all of within[i:i+w] are True.
                        # Only scan from after peak time to avoid pre-peak baseline.
                        scan_start = peak_offset + 1 if peak_offset < len(chrg_seg) else 0
                        for i in range(scan_start, len(chrg_seg) - ret_dur_samples + 1):
                            if within[i:i + ret_dur_samples].all():
                                cutoff = i
                                break
                    charge_pC = float(np.trapz(chrg_seg[:cutoff], dx=dt))
                elif charge_end > p_start:
                    # No sigma → just integrate the whole window
                    chrg_seg = cur[p_start:charge_end] - local_base
                    charge_pC = float(np.trapz(chrg_seg, dx=dt))

                row = {
                    'sweep': sweep_num,
                    'pulse_num': p_i + 1,
                    'pulse_onset_s': p_start / sr,
                    'baseline_pA': local_base,
                    'pre_pulse_baseline_pA': baseline,
                    'pre_pulse_sigma_MAD_pA': sigma_mad,
                    'peak_pA': peak_val,
                    'peak_time_ms': peak_time_ms,
                    'latency_ms': latency_ms,
                    'rise_2080_ms': rise_2080_ms,
                    'decay_tau_ms': decay_tau_ms,
                    'charge_pC': charge_pC,
                    'is_responder': is_responder,
                    'n_pulses_in_sweep': n_pulses,
                }
                if label_col_name is not None:
                    if sweep_to_label_map is not None:
                        if sweep_num in sweep_to_label_map:
                            row[label_col_name] = sweep_to_label_map[sweep_num]
                        else:
                            row[label_col_name] = np.nan
                            print(f"  [warn] cell {self.name}: sweep {sweep_num} "
                                  f"falls outside any range in sweep_to_label "
                                  f"({label_col_name}); tagging NaN")
                    else:
                        row[label_col_name] = pulse_label[1]
                rows.append(row)
                peaks_signed.append((peak_val, is_responder))

            # ---- PPR for this sweep (peak2 / peak1) ----
            # Tag PPR onto the pulse-2 row (NaN if pulse 1 wasn't a responder)
            if len(peaks_signed) >= 2 and len(rows) >= 2:
                p1_val, p1_resp = peaks_signed[0]
                p2_val, _ = peaks_signed[1]
                if p1_resp and abs(p1_val) > 0:
                    ppr = p2_val / p1_val
                else:
                    ppr = np.nan
                rows[-1]['PPR_2vs1'] = ppr  # tag on most recent (pulse 2) row

        new_rows_df = pd.DataFrame(rows)
        # Accumulate across calls (one .h5 file at a time): append to any existing
        # optoSweeps rather than overwriting. The grouped table is then recomputed
        # from the full accumulated table so groupings are correct.
        if hasattr(self, 'optoSweeps') and self.optoSweeps is not None \
                and not self.optoSweeps.empty:
            self.optoSweeps = pd.concat([self.optoSweeps, new_rows_df], ignore_index=True)
        else:
            self.optoSweeps = new_rows_df

        # ---- Grouped table: responder-only means, with n_used/n_total counts ----
        if self.optoSweeps.empty:
            self.optoGrouped = pd.DataFrame()
            return

        df = self.optoSweeps.copy()
        # Group keys: pulse_num + n_pulses_in_sweep (so single-pulse and PPR
        # don't collapse together), plus any per-sweep label columns.
        group_keys = ['pulse_num', 'n_pulses_in_sweep']
        candidate_label_cols = [c for c in df.columns
                                if c not in {'sweep', 'pulse_num', 'pulse_onset_s',
                                             'baseline_pA', 'pre_pulse_baseline_pA',
                                             'pre_pulse_sigma_MAD_pA', 'peak_pA',
                                             'peak_time_ms', 'latency_ms',
                                             'rise_2080_ms', 'decay_tau_ms',
                                             'charge_pC', 'is_responder',
                                             'n_pulses_in_sweep', 'PPR_2vs1'}]
        if candidate_label_cols:
            group_keys.extend(candidate_label_cols)

        # Responder-only means for amplitude / kinetics
        metric_cols = ['peak_pA', 'peak_time_ms', 'latency_ms',
                       'rise_2080_ms', 'decay_tau_ms', 'charge_pC']
        df_resp = df[df['is_responder']]
        means = (df_resp.groupby(group_keys, dropna=False)[metric_cols]
                        .mean()
                        .reset_index())
        n_used = (df_resp.groupby(group_keys, dropna=False)
                          .size().reset_index(name='n_sweeps_used'))
        n_total = (df.groupby(group_keys, dropna=False)
                      .size().reset_index(name='n_sweeps_total'))

        grouped = means.merge(n_used, on=group_keys, how='outer') \
                       .merge(n_total, on=group_keys, how='outer')
        grouped['n_sweeps_used']  = grouped['n_sweeps_used'].fillna(0).astype(int)
        grouped['n_sweeps_total'] = grouped['n_sweeps_total'].fillna(0).astype(int)

        # PPR mean: only meaningful on pulse_num == 2 rows
        if 'PPR_2vs1' in df.columns:
            ppr_mean = (df.dropna(subset=['PPR_2vs1'])
                          .groupby(group_keys, dropna=False)['PPR_2vs1']
                          .mean()
                          .reset_index()
                          .rename(columns={'PPR_2vs1': 'PPR_mean'}))
            grouped = grouped.merge(ppr_mean, on=group_keys, how='left')

        self.optoGrouped = grouped

    def buildOptoTemplates(self, params=None):
        """
        Two-pass responder classification using per-intensity averaging + cross-correlation.

        Pass 1 — Build per-intensity templates:
            Average all sweeps at each intensity in a 220ms window (20ms pre-pulse
            + 200ms post-pulse). Assess template SNR (peak/RMS of pre-pulse segment).
            If SNR < snr_min, borrow template from nearest higher intensity that passes.
            If no intensity passes, fall back to amplitude-only GMM classification.

        Pass 2 — Cross-correlate each sweep against its template:
            Compute Pearson r between template window and same window in each sweep.
            Threshold via GMM on the score distribution (valley between failure and
            success clusters). Falls back to mean + n_sigma_fallback * SD if GMM
            components are too close together.

        Adds to cell.optoSweeps:
            corr_score           Pearson r against template
            template_intensity   which intensity's template was used
            is_responder_xcorr   bool classification

        Rebuilds cell.optoGrouped using is_responder_xcorr.

        Optional params dict keys (all have defaults):
            sampling_rate          (10000)
            template_pre_s         (0.020)    baseline window before pulse in template
            snr_min                (3.0)      min peak/RMS to use a template
            n_sigma_fallback       (2.0)      fallback threshold sigma multiplier
            latency_min_s          (0.001)    ignore artifact before this post-pulse
            gmm_min_separation     (0.15)     min component separation to trust GMM
        """
        from sklearn.mixture import GaussianMixture

        if not hasattr(self, 'optoSweeps') or self.optoSweeps is None \
                or self.optoSweeps.empty:
            print(f'  [buildOptoTemplates] {self.name}: no optoSweeps, skipping')
            return

        if params is None:
            params = {}

        sr          = int(params.get('sampling_rate',        10000))
        pre_s       = float(params.get('template_pre_s',     0.020))
        snr_min     = float(params.get('snr_min',            3.0))
        n_sig_fb    = float(params.get('n_sigma_fallback',   2.0))
        lat_min_s   = float(params.get('latency_min_s',      0.001))
        gmm_min_sep = float(params.get('gmm_min_separation', 0.15))

        pre_samps = int(round(pre_s * sr))
        lat_samps = int(round(lat_min_s * sr))

        df  = self.optoSweeps.copy()
        df1 = df[(df['pulse_num'] == 1) & (df['n_pulses_in_sweep'] == 1)].copy()

        if not hasattr(self, 'optoRawWindows') or not self.optoRawWindows:
            print(f'  [buildOptoTemplates] {self.name}: no raw windows — '
                  f'using amplitude-only fallback (re-run ephysAnalysisCreate to get xcorr)')
            self._buildOptoTemplates_amplitudeOnly(df, snr_min, n_sig_fb, gmm_min_sep)
            return

        if df1.empty:
            print(f'  [buildOptoTemplates] {self.name}: no single-pulse sweeps, skipping')
            return

        raw = self.optoRawWindows

        # intensity column is only present when sweep_to_label / pulse_label was set
        if 'intensity' not in df1.columns:
            df1 = df1.copy()
            df1['intensity'] = 'all'

        intensities = sorted(df1['intensity'].dropna().unique())
        templates   = {}
        snrs        = {}

        # ── Pass 1: build per-intensity templates ─────────────────────────────
        for intens in intensities:
            sweep_nums = df1.loc[df1['intensity'] == intens, 'sweep'].values
            windows    = [raw[s] for s in sweep_nums if s in raw]
            if not windows:
                templates[intens] = None
                snrs[intens]      = 0.0
                continue
            avg      = np.mean(np.vstack(windows), axis=0)
            pre_seg  = avg[:pre_samps]
            post_seg = avg[pre_samps + lat_samps:]
            noise    = np.sqrt(np.mean(pre_seg ** 2)) if len(pre_seg) > 0 else 1.0
            signal   = np.abs(post_seg).max() if len(post_seg) > 0 else 0.0
            snr      = signal / noise if noise > 0 else 0.0
            templates[intens] = avg
            snrs[intens]      = snr

        # ── Fallback: borrow from nearest higher-SNR intensity ────────────────
        template_source = {}
        sorted_intens   = sorted(intensities)
        for intens in sorted_intens:
            if snrs.get(intens, 0) >= snr_min:
                template_source[intens] = intens
            else:
                candidates = [i for i in sorted_intens
                              if i > intens and snrs.get(i, 0) >= snr_min]
                if candidates:
                    donor = candidates[0]
                else:
                    any_good = [i for i in sorted_intens if snrs.get(i, 0) >= snr_min]
                    donor    = any_good[-1] if any_good else None

                if donor is not None:
                    templates[intens]       = templates[donor]
                    template_source[intens] = donor
                    print(f'  [buildOptoTemplates] {self.name} intensity {intens}: '
                          f'SNR {snrs.get(intens, 0):.1f} < {snr_min}, '
                          f'borrowing from {donor} (SNR {snrs.get(donor, 0):.1f})')
                else:
                    template_source[intens] = None
                    print(f'  [buildOptoTemplates] {self.name} intensity {intens}: '
                          f'no usable template')

        self.template_source = template_source

        # ── Pass 2: cross-correlate each sweep against its template ───────────
        def pearson_r(a, b):
            a     = a - a.mean()
            b     = b - b.mean()
            denom = np.sqrt((a ** 2).sum() * (b ** 2).sum())
            return float(np.dot(a, b) / denom) if denom > 0 else 0.0

        corr_scores      = {}
        tmpl_intens_used = {}

        for _, row in df1.iterrows():
            sw     = int(row['sweep'])
            intens = row.get('intensity', 'all')
            tmpl   = templates.get(intens)
            src    = template_source.get(intens)

            if tmpl is None or sw not in raw:
                corr_scores[sw]      = np.nan
                tmpl_intens_used[sw] = np.nan
                continue

            sweep_win = raw[sw]
            n         = min(len(tmpl), len(sweep_win))
            corr_scores[sw]      = pearson_r(tmpl[:n], sweep_win[:n])
            tmpl_intens_used[sw] = src

        # ── Threshold: GMM, fallback to sigma ─────────────────────────────────
        valid_scores = np.array([v for v in corr_scores.values()
                                 if np.isfinite(v)]).reshape(-1, 1)
        threshold = np.nan

        if len(valid_scores) >= 6:
            try:
                gmm   = GaussianMixture(n_components=2, random_state=42, max_iter=200)
                gmm.fit(valid_scores)
                means = gmm.means_.flatten()
                sep   = abs(means[0] - means[1])
                if sep >= gmm_min_sep:
                    lo, hi = sorted(means)
                    scan   = np.linspace(lo, hi, 200)
                    probs  = gmm.predict_proba(scan.reshape(-1, 1))
                    diff   = np.abs(probs[:, 0] - probs[:, 1])
                    threshold = float(scan[np.argmin(diff)])
                    print(f'  [buildOptoTemplates] {self.name}: GMM threshold = '
                          f'{threshold:.3f} (separation {sep:.3f})')
                else:
                    print(f'  [buildOptoTemplates] {self.name}: GMM components '
                          f'too close ({sep:.3f}), using sigma fallback')
            except Exception as e:
                print(f'  [buildOptoTemplates] {self.name}: GMM failed ({e}), '
                      f'using sigma fallback')

        if np.isnan(threshold):
            mu        = float(np.mean(valid_scores))
            sigma     = float(np.std(valid_scores))
            threshold = mu + n_sig_fb * sigma
            print(f'  [buildOptoTemplates] {self.name}: sigma fallback '
                  f'threshold = {threshold:.3f}')

        # ── Tag results onto optoSweeps ───────────────────────────────────────
        df['corr_score']         = df['sweep'].map(corr_scores)
        df['template_intensity'] = df['sweep'].map(tmpl_intens_used)
        df['is_responder_xcorr'] = df['corr_score'] > threshold

        # Carry pulse-1 xcorr classification forward to pulse-2 rows
        for sw, grp in df.groupby('sweep'):
            if grp['pulse_num'].max() > 1:
                p1_resp = grp.loc[grp['pulse_num'] == 1, 'is_responder_xcorr']
                if not p1_resp.empty:
                    df.loc[(df['sweep'] == sw) & (df['pulse_num'] > 1),
                           'is_responder_xcorr'] = p1_resp.values[0]

        self.optoSweeps      = df
        self.xcorr_threshold = threshold
        self.template_snrs   = snrs
        self.templates       = templates

        self._recomputeOptoGrouped(responder_col='is_responder_xcorr')

    def _buildOptoTemplates_amplitudeOnly(self, df, snr_min, n_sig_fb, gmm_min_sep):
        """Fallback amplitude-only GMM classification when raw windows unavailable."""
        from sklearn.mixture import GaussianMixture

        df1          = df[(df['pulse_num'] == 1) & (df['n_pulses_in_sweep'] == 1)].copy()
        if 'intensity' not in df1.columns:
            df1['intensity'] = 'all'
        intensities  = sorted(df1['intensity'].dropna().unique())
        global_sigma = df1['pre_pulse_sigma_MAD_pA'].median()
        thresholds   = {}

        for intens in intensities:
            amps = df1.loc[df1['intensity'] == intens, 'peak_pA'].abs().values
            if len(amps) < 4:
                thresholds[intens] = n_sig_fb * global_sigma
                continue
            try:
                gmm   = GaussianMixture(n_components=2, random_state=42)
                gmm.fit(amps.reshape(-1, 1))
                means = gmm.means_.flatten()
                sep   = abs(means[0] - means[1])
                if sep >= gmm_min_sep * amps.max():
                    scan  = np.linspace(min(means), max(means), 200)
                    probs = gmm.predict_proba(scan.reshape(-1, 1))
                    diff  = np.abs(probs[:, 0] - probs[:, 1])
                    thresholds[intens] = float(scan[np.argmin(diff)])
                else:
                    thresholds[intens] = n_sig_fb * global_sigma
            except Exception:
                thresholds[intens] = n_sig_fb * global_sigma

        df = self.optoSweeps.copy()

        def classify(row):
            if row['pulse_num'] != 1:
                return np.nan
            thresh = thresholds.get(row.get('intensity'), n_sig_fb * global_sigma)
            return bool(abs(row['peak_pA']) > thresh)

        df['corr_score']         = np.nan
        df['template_intensity'] = np.nan
        df['is_responder_xcorr'] = df.apply(classify, axis=1)
        self.optoSweeps          = df
        self._recomputeOptoGrouped(responder_col='is_responder_xcorr')

    def _recomputeOptoGrouped(self, responder_col='is_responder_xcorr'):
        """Rebuild optoGrouped using the specified responder column."""
        df         = self.optoSweeps.copy()
        group_keys = ['pulse_num', 'n_pulses_in_sweep']
        candidate_label_cols = [c for c in df.columns
                                if c not in {'sweep', 'pulse_num', 'pulse_onset_s',
                                             'baseline_pA', 'pre_pulse_baseline_pA',
                                             'pre_pulse_sigma_MAD_pA', 'peak_pA',
                                             'peak_time_ms', 'latency_ms',
                                             'rise_2080_ms', 'decay_tau_ms',
                                             'charge_pC', 'is_responder',
                                             'is_responder_xcorr', 'corr_score',
                                             'template_intensity',
                                             'n_pulses_in_sweep', 'PPR_2vs1'}]
        if candidate_label_cols:
            group_keys.extend(candidate_label_cols)

        metric_cols = ['peak_pA', 'peak_time_ms', 'latency_ms',
                       'rise_2080_ms', 'decay_tau_ms', 'charge_pC']
        df_resp = df[df[responder_col] == True]
        means   = df_resp.groupby(group_keys, dropna=False)[metric_cols].mean().reset_index()
        n_used  = df_resp.groupby(group_keys, dropna=False).size().reset_index(name='n_sweeps_used')
        n_total = df.groupby(group_keys, dropna=False).size().reset_index(name='n_sweeps_total')

        grouped = means.merge(n_used,  on=group_keys, how='outer') \
                       .merge(n_total, on=group_keys, how='outer')
        grouped['n_sweeps_used']  = grouped['n_sweeps_used'].fillna(0).astype(int)
        grouped['n_sweeps_total'] = grouped['n_sweeps_total'].fillna(0).astype(int)

        if 'PPR_2vs1' in df.columns:
            ppr_mean = (df.dropna(subset=['PPR_2vs1'])
                          .groupby(group_keys, dropna=False)['PPR_2vs1']
                          .mean().reset_index()
                          .rename(columns={'PPR_2vs1': 'PPR_mean'}))
            grouped = grouped.merge(ppr_mean, on=group_keys, how='left')

        self.optoGrouped = grouped

    def addMCurrent(self, data, params):
        """
        Quantify M-type K+ current (Kv7/KCNQ) from a VC hyperpolarizing-step
        deactivation protocol.

        Protocol (JN_Ch1_VC_mCurrent_hyperpolStep.cfg): cell held at -10 mV.
        Each sweep delivers a 10 ms VC test pulse at sweep start, then a 1 s
        hyperpolarizing step. Step amplitude is a per-sweep delta from the
        holding potential (-60, -50, ... 0 mV across 7 sweeps), so absolute
        step voltages are -70 ... -10 mV at a -10 mV hold. During the step,
        M-channels deactivate (slow inward relaxation); on step-back they
        reactivate (slow tail).

        Step voltage is computed from each sweep's POSITION in the file, not
        read from the recorded command-voltage channel — that channel can
        carry a scale error and only records the stimulus delta. The 7 sweeps
        in one .h5 are always the 7 steps in order regardless of their
        absolute sweep numbers.

        Per sweep, computes:
            position               1-based sweep position within the file
            step_voltage_mV        computed: holding_mV + step_delta_mV *
                                   (position - zero_step_position)
            holding_current_pA     mean current during pre-step hold
            inst_current_pA        current just after the capacitive transient
                                   settles at step onset (M still at hold level)
            ss_current_pA          mean current over the last part of the step
                                   (M fully deactivated)
            relaxation_pA          inst_current - ss_current; the M-current that
                                   deactivated during the step
            deact_tau_ms           single-exp fit to the deactivation relaxation
            tail_amp_pA            reactivation tail amplitude on step-back
            tail_tau_ms            single-exp fit to the reactivation tail
            conductance_nS         relaxation_pA / (step_voltage_mV - E_K)
            *_per_pF               Normalized current density utilizing the 
                                   cell's mean capacitance.

        Stores per-sweep table as cell.mCurrentSweeps (DataFrame) and a
        per-step-voltage grouped mean as cell.mCurrentGrouped (DataFrame).
        Accumulates across calls (one .h5 file at a time).

        Required params attributes (defaults in parens):
            sampling_rate              (10000)
            current_channel_index      (0)     analog channel with current
            holding_mV                 (-10.0) holding potential
            step_delta_mV              (10.0)  command delta per sweep position
            zero_step_position         (7)     position whose delta is zero
            step_onset_s               (0.2)   hyperpolarizing step start
            step_duration_s            (1.0)   step length
            cap_settle_s               (0.01)  skip this long after step onset
                                               before measuring inst_current
            inst_window_s              (0.005) inst_current averaging window
            ss_fraction                (0.1)   ss_current = last this-fraction
                                               of the step
            hold_measure_window_s      (0.05)  pre-step holding-current window
            hold_measure_gap_s         (0.01)  end hold window this far before step
            tail_window_s              (0.3)   reactivation tail fit window after
                                               step-back
            deact_tau_min_s            (0.005) deactivation fit bounds: 5 ms
            deact_tau_max_s            (1.0)   deactivation fit bounds: 1 s
            E_K_mV                     (-97.0) K reversal for conductance calc
        """
        sr = float(getattr(params, 'sampling_rate', 10000))
        dt = 1.0 / sr
        cur_idx = int(getattr(params, 'current_channel_index', 0))
        step_onset_s = float(getattr(params, 'step_onset_s', 0.2))
        step_dur_s = float(getattr(params, 'step_duration_s', 1.0))
        cap_settle_s = float(getattr(params, 'cap_settle_s', 0.01))
        inst_win_s = float(getattr(params, 'inst_window_s', 0.005))
        ss_frac = float(getattr(params, 'ss_fraction', 0.1))
        hold_win_s = float(getattr(params, 'hold_measure_window_s', 0.05))
        hold_gap_s = float(getattr(params, 'hold_measure_gap_s', 0.01))
        tail_win_s = float(getattr(params, 'tail_window_s', 0.3))
        tau_min = float(getattr(params, 'deact_tau_min_s', 0.005))
        tau_max = float(getattr(params, 'deact_tau_max_s', 1.0))
        E_K = float(getattr(params, 'E_K_mV', -97.0))
        
        # Step voltage is computed from sweep POSITION in the file, not read
        # from the recorded command channel (that channel can carry a scale
        # error and only records the stimulus delta, not absolute mV).
        holding_mV = float(getattr(params, 'holding_mV', -10.0))
        step_delta_mV = float(getattr(params, 'step_delta_mV', 10.0))
        zero_step_position = int(getattr(params, 'zero_step_position', 7))

        # Mean capacitance for the cell (self.capacitance is already in pF)
        mean_cap_pF = np.nan
        if hasattr(self, 'capacitance') and self.capacitance:
            valid_caps = [v for v in self.capacitance.values() if np.isfinite(v)]
            if valid_caps:
                mean_cap_pF = np.mean(valid_caps)

        def exp_decay(t, A, tau, C):
            return A * np.exp(-t / tau) + C

        # Sample-index landmarks
        step_start = int(round(step_onset_s * sr))
        step_end = int(round((step_onset_s + step_dur_s) * sr))
        cap_settle = int(round(cap_settle_s * sr))
        inst_win = int(round(inst_win_s * sr))
        hold_win = int(round(hold_win_s * sr))
        hold_gap = int(round(hold_gap_s * sr))
        tail_win = int(round(tail_win_s * sr))

        sweepsList = {k: v for k, v in data.items() if 'sweep' in k}
        if not sweepsList:
            return

        # Sort sweeps by sweep number so file position is deterministic
        sorted_sweep_names = sorted(sweepsList.keys(),
                                    key=lambda nm: self.stripSweep(nm))

        rows = []
        for position, sweep_name in enumerate(sorted_sweep_names, start=1):
            sweep_num = self.stripSweep(sweep_name)
            scans = np.transpose(np.array(sweepsList[sweep_name]['analogScans']))
            n_ch, n_samp = scans.shape
            cur = scans[min(cur_idx, n_ch - 1)].astype(float)

            if step_end > n_samp:
                # trace too short for this protocol; skip sweep
                continue

            # Step voltage computed from sweep position in the file
            step_voltage = holding_mV + step_delta_mV * (position - zero_step_position)

            # Holding current: pre-step window ending hold_gap before the step
            hold_end = max(0, step_start - hold_gap)
            hold_start = max(0, hold_end - hold_win)
            holding_current = float(np.mean(cur[hold_start:hold_end])) \
                if hold_end > hold_start else np.nan

            # Instantaneous current at step onset (after capacitive transient)
            inst_start = step_start + cap_settle
            inst_end = min(inst_start + inst_win, step_end)
            inst_current = float(np.mean(cur[inst_start:inst_end])) \
                if inst_end > inst_start else np.nan

            # Steady-state current: last ss_frac of the step
            ss_start = max(inst_end, step_end - int(round(ss_frac * step_dur_s * sr)))
            ss_current = float(np.mean(cur[ss_start:step_end])) \
                if step_end > ss_start else np.nan

            # M-current relaxation amplitude
            relaxation = inst_current - ss_current \
                if (np.isfinite(inst_current) and np.isfinite(ss_current)) else np.nan

            # Deactivation tau: single-exp fit to the relaxation phase
            deact_tau_ms = np.nan
            deact_seg = cur[inst_start:step_end]
            if len(deact_seg) > 5 and np.isfinite(relaxation):
                t_axis = np.arange(len(deact_seg)) * dt
                p0 = (float(deact_seg[0] - deact_seg[-1]), 0.1, float(deact_seg[-1]))
                bounds = ([-np.inf, tau_min, -np.inf], [np.inf, tau_max, np.inf])
                try:
                    popt, _ = curve_fit(exp_decay, t_axis, deact_seg,
                                        p0=p0, bounds=bounds, maxfev=5000)
                    tau_s = popt[1]
                    if tau_min * 1.01 < tau_s < tau_max * 0.99:
                        deact_tau_ms = tau_s * 1000.0
                except Exception:
                    pass

            # Reactivation tail on step-back at step_end
            tail_amp = np.nan
            tail_tau_ms = np.nan
            tail_end = min(step_end + tail_win, n_samp)
            if tail_end - step_end > 5:
                # skip a few samples for the capacitive transient on step-back
                tail_settle = step_end + cap_settle
                tail_seg = cur[tail_settle:tail_end]
                if len(tail_seg) > 5:
                    # tail amplitude: first settled sample minus final sample
                    tail_amp = float(tail_seg[0] - tail_seg[-1])
                    t_axis = np.arange(len(tail_seg)) * dt
                    p0 = (float(tail_seg[0] - tail_seg[-1]), 0.1, float(tail_seg[-1]))
                    bounds = ([-np.inf, tau_min, -np.inf], [np.inf, tau_max, np.inf])
                    try:
                        popt, _ = curve_fit(exp_decay, t_axis, tail_seg,
                                            p0=p0, bounds=bounds, maxfev=5000)
                        tau_s = popt[1]
                        if tau_min * 1.01 < tau_s < tau_max * 0.99:
                            tail_tau_ms = tau_s * 1000.0
                    except Exception:
                        pass

            # Conductance from the deactivating current: G = I / (V - E_K)
            driving_force = step_voltage - E_K
            conductance_nS = (relaxation / driving_force) \
                if (np.isfinite(relaxation) and np.isfinite(driving_force)
                    and abs(driving_force) > 1e-9) else np.nan

            rows.append({
                'sweep': sweep_num,
                'position': position,
                'step_voltage_mV': step_voltage,
                'holding_current_pA': holding_current,
                'inst_current_pA': inst_current,
                'ss_current_pA': ss_current,
                'relaxation_pA': relaxation,
                'deact_tau_ms': deact_tau_ms,
                'tail_amp_pA': tail_amp,
                'tail_tau_ms': tail_tau_ms,
                'conductance_nS': conductance_nS,
                'cell_capacitance_pF': mean_cap_pF,
                'relaxation_pA_per_pF': relaxation / mean_cap_pF if mean_cap_pF > 0 else np.nan,
                'holding_current_pA_per_pF': holding_current / mean_cap_pF if mean_cap_pF > 0 else np.nan,
                'inst_current_pA_per_pF': inst_current / mean_cap_pF if mean_cap_pF > 0 else np.nan,
                'ss_current_pA_per_pF': ss_current / mean_cap_pF if mean_cap_pF > 0 else np.nan,
                'tail_amp_pA_per_pF': tail_amp / mean_cap_pF if mean_cap_pF > 0 else np.nan,
            })

        new_rows_df = pd.DataFrame(rows)
        # Accumulate across calls (one .h5 file at a time)
        if hasattr(self, 'mCurrentSweeps') and self.mCurrentSweeps is not None \
                and not self.mCurrentSweeps.empty:
            self.mCurrentSweeps = pd.concat([self.mCurrentSweeps, new_rows_df],
                                            ignore_index=True)
        else:
            self.mCurrentSweeps = new_rows_df

        if self.mCurrentSweeps.empty:
            self.mCurrentGrouped = pd.DataFrame()
            return

        # Grouped: mean per step voltage. Round the voltage to the nearest mV
        # so tiny command-voltage noise doesn't fragment the groups.
        df = self.mCurrentSweeps.copy()
        df['step_voltage_mV_rounded'] = df['step_voltage_mV'].round(0)
        
        metric_cols = ['step_voltage_mV', 'holding_current_pA', 'inst_current_pA',
                       'ss_current_pA', 'relaxation_pA', 'deact_tau_ms',
                       'tail_amp_pA', 'tail_tau_ms', 'conductance_nS',
                       'cell_capacitance_pF', 'relaxation_pA_per_pF',
                       'holding_current_pA_per_pF', 'inst_current_pA_per_pF',
                       'ss_current_pA_per_pF', 'tail_amp_pA_per_pF']
                       
        grouped = (df.groupby('step_voltage_mV_rounded', dropna=False)[metric_cols]
                     .mean().reset_index())
        n_sweeps = (df.groupby('step_voltage_mV_rounded', dropna=False)
                      .size().reset_index(name='n_sweeps'))
        grouped = grouped.merge(n_sweeps, on='step_voltage_mV_rounded', how='outer')
        self.mCurrentGrouped = grouped

        
    def addRMP(self, data):
        """
        addRMP _summary_

        _extended_summary_

        Args:
            data (_type_): _description_
        """
        if not hasattr(self, 'RMPData'):
            self.RMPData = {}

        sweepsList = {key: value for key,
                      value in data.items() if 'sweep' in key}
        for sweep in sweepsList:
            sweepNum = self.stripSweep(sweep)
            trace = np.transpose(np.array(data[sweep]['analogScans']))[0]
            if np.mean(trace) > -20:
                pass
            else:
                self.RMPData[sweepNum] = np.mean(trace)

    def addRheobase(self, data, params):
        if not hasattr(self, 'rheobase'):
            self.rheobase = {}

        def findSpike(trace, params):
            spikes, spikeProps = scipy.signal.find_peaks(
                trace, height=params.threshold, distance=(0.005*params.sampling_rate))
            if len(spikes) == 0:
                return False
            elif len(spikes) > 0:
                return True

        traces = [np.transpose(np.array(data[sweep]['analogScans']))[
            0] for sweep in self.rheoCurrentDict.keys()]
        spikeBoolList = list(
            map(lambda trace: findSpike(trace, params), traces))
        sweepNums = list(map(lambda sweep: self.stripSweep(
            sweep), list(self.rheoCurrentDict.keys())))

        rheobaseVal = next((current for spikeBool, current, sweepNum in zip(
            spikeBoolList, self.rheoCurrentDict.values(), sweepNums) if spikeBool), None)
        rheobaseSweep = next((sweepNum for spikeBool, current, sweepNum in zip(
            spikeBoolList, self.rheoCurrentDict.values(), sweepNums) if spikeBool), None)

        if rheobaseSweep is None:
            # Fired on no sweep: censor at highest current tested + one step.
            censored = censored_rheobase_value(self.rheoCurrentDict)
            if censored is not None:
                rheobaseVal = censored
                rheobaseSweep = max(sweepNums)   # the highest-current sweep

        self.rheobase[rheobaseSweep] = rheobaseVal

    def addRheobaseFromFI(self, data, params):
        """Fallback rheobase from the F-I (20 pA) protocol.

        The dedicated 5 pA rheobase protocol injects current for only 50 ms,
        whereas the F-I protocol injects for 500 ms. To keep the fallback
        comparable to the real rheobase measure, spikes are only counted in
        the FIRST 50 ms of the F-I current step. Stores the lowest current
        whose sweep fires >=1 spike within that window in self.fiRheobase
        (a {sweep: current} dict, mirroring self.rheobase). Spike detection
        is identical to addRheobase (scipy.signal.find_peaks, height=
        params.threshold). Computed for every cell with an F-I recording;
        it is only USED as the rheobase value for cells that have no real
        5 pA rheobase (see ephysAnalysisAnalyze._resolve_rheobase_with_FI_fallback).

        Required params: sampling_rate, threshold, delay (current-step onset, s).
        """
        if not hasattr(self, 'fiRheobase'):
            self.fiRheobase = {}
        if not getattr(self, 'FRCurrentDict', None):
            return

        sr = float(params.sampling_rate)
        onset = int(round(float(params.delay) * sr))      # current-step onset
        window_end = onset + int(round(0.050 * sr))       # first 50 ms only

        def firesInFirst50ms(trace):
            seg = trace[onset:window_end]
            spikes, _ = scipy.signal.find_peaks(
                seg, height=params.threshold, distance=(0.005 * sr))
            return len(spikes) > 0

        # Walk current steps low -> high; first one that fires is the rheobase.
        for sweepName, current in sorted(self.FRCurrentDict.items(),
                                         key=lambda kv: kv[1]):
            trace = np.transpose(np.array(data[sweepName]['analogScans']))[0]
            if firesInFirst50ms(trace):
                self.fiRheobase[self.stripSweep(sweepName)] = current
                break

    def createSpikeAnalysis(self, data, params, currentDictVar, protocol):
        # spikeAnalysis = {}
        keys_for_dict = ['current injected', 'protocol', 'sweep','threshold_v',
                        'upstroke', 'peak_v','trough_v','upstroke_v','downstroke',
                        'downstroke_v','width', 'adapt','latency','isi_cv','mean_isi',
                        'median_isi', 'first_isi','phasePlotData', 'rawSpikes', 'rawTrain']
        
        if not hasattr(self, 'spikeAnalysis'):
            self.spikeAnalysis = {}
            for key in keys_for_dict:
                self.spikeAnalysis[key] = []

        def currentAmpTS(currentInjection, delay, duration, end, sampleRate):
                    #* convert timings to sample rate
            delayCon = int(float(delay)*sampleRate)
            durationCon = int(float(duration)*sampleRate)
            endCon = int(float(end)*sampleRate)
            # total_time = float(delay) + float(duration) + float(end)

            #* initialize the boi
            currentTS = np.zeros(int(delayCon+durationCon+endCon))
            # timestamps = np.linspace(0, total_time, int(total_time*sampleRate))
            #* insert current injection
            currentTS[delayCon+1:delayCon+durationCon] = currentInjection
            return currentTS
        
        def isolateAP(data, timestamps, idx, window):
            if idx < 50:
                dV_dt = np.nan
                spikeTrace = np.nan
            else:
                spikeTrace = data[idx-window:idx+window]
                spikeTime = timestamps[idx-window:idx+window]
                dV_dt = np.gradient(spikeTrace,spikeTime)

            return dV_dt, spikeTrace

        sweepsList = {key: value for key,
                      value in data.items() if 'sweep' in key}

        stripped_sweeps = map(lambda sweep: self.stripSweep(
            sweep), list(sweepsList.keys()))
        # for sweep in sweepsList:
        traces = [np.transpose(np.array(data[sweep]['analogScans']))[0] for sweep in getattr(self, currentDictVar).keys()]
        currentAmps = list(getattr(self, currentDictVar).values())
        
        # currentInjs = list(map(lambda currentNum: currentAmpTS(
        #     currentNum, params.delay, params.duration, params.end, params.sampleRate), currentAmps))
        
        for voltageData, currentNumber, sweep in zip(traces, currentAmps, stripped_sweeps):
            currentInj = currentAmpTS(currentNumber, params.delay, params.stimDuration, params.end, params.sampling_rate)
            ext = SpikeFeatureExtractor()
            total_time = len(voltageData)/params.sampling_rate
            
            timestamps = np.linspace(0, total_time, len(voltageData))
            if len(currentInj) < len(voltageData):
                currentInj = np.pad(currentInj, (0, len(voltageData) - len(currentInj)), mode='constant', constant_values=0)

            
            spikes = ext.process(timestamps, voltageData, currentInj)
            # %%
            ext1 = SpikeTrainFeatureExtractor(start=None, end=None)

            features = ext1.process(timestamps, voltageData, currentInj, spikes) # re-using spikes from above
            rawTrace = voltageData.copy()

            # phasePlotList = []
            if spikes.empty == False:
                spikes['phasePlotData'] = spikes['peak_index'].apply(lambda idx: isolateAP(voltageData, timestamps, idx, 50))
                spikes['rawSpikes'] = spikes['peak_index'].apply(lambda idx: voltageData[idx-3000:idx+3000])
            

            for boi in keys_for_dict:

                if boi == 'current injected':
                    self.spikeAnalysis['current injected'].append(currentNumber)
                elif boi == 'protocol':
                    self.spikeAnalysis['protocol'].append(protocol)
                elif boi == 'sweep':
                    self.spikeAnalysis['sweep'].append(sweep)
                elif boi == 'rawTrain':
                    # Store the full sweep trace only when the sweep has spikes,
                    # consistent with rawSpikes/phasePlotData above. Spikeless
                    # sweeps' traces are never read downstream and bloated the
                    # collective pickle by ~0.7 GB.
                    self.spikeAnalysis['rawTrain'].append(
                        rawTrace if not spikes.empty else np.nan)
                elif boi in spikes.columns:
                    self.spikeAnalysis[boi].append(list(spikes[boi]))
                elif boi in features.keys():
                    self.spikeAnalysis[boi].append(features[boi])
                else:
                    self.spikeAnalysis[boi].append(np.nan)
    # def phasePlotAnalysis(self, data):
    #     a=1
    #     pass           

class ProtocolParams:
    def __init__(self, name, **kwargs):
        self.name = name
        for key, value in kwargs.items():
            setattr(self, key, value)


# %%

#! probably not gonna use this class
# class EphysSweep:
#     def __init__(self, protocol, holdingSweeps):
#         self.protocol = protocol
#         self.holdingSweeps = holdingSweeps

#     # def findFiringRate(trace,threshold, stimDuration,sampling_rate):
#     #     spikes, spikeProps = scipy.signal.find_peaks(trace,height=threshold, distance=(0.005*sampling_rate))
#     #     spikeRate = len(spikes)/float(stimDuration)
#     #     return spikeRate, spikeProps

#     def runAnalysis(self):
#         if self.protocol == 'PA_Ch1_CC Ih -50 pA steps to -150 pA.cfg':
#             pass
#         elif self.protocol == "PA_Ch1_CC excitability 20 pA steps to 200 pA.cfg":
#             self.data = findFiringRate(self)


def makeMice():
    m005 = EphysMouse('610-005', 'naive', 'female', 'SNL', False)
    m006 = EphysMouse('611-006', 'stress', 'male', 'SNL', True)
    m009 = EphysMouse('607-009', 'stress', 'male', 'SNL', True)
    m022 = EphysMouse('607-022', 'stress', 'male', 'SNL')
    m023 = EphysMouse('614-023', 'naive', 'female', 'SNL')
    m323 = EphysMouse('611-323', 'stress', 'male', 'SNL')
    m011 = EphysMouse('608-011', 'naive', 'male', 'SNL')
    m021 = EphysMouse('608-021', 'naive', 'male', 'SNL')
    m001 = EphysMouse('674-R', 'stress', 'female', 'BNST')
    m101 = EphysMouse('671-L', 'stress', 'male', 'PAG')
    m102 = EphysMouse('675-RL', 'naive', 'female', 'BNST')
    m103 = EphysMouse('675-L', 'naive', 'female', 'PAG')
    m104 = EphysMouse('671-0', 'stress', 'male', 'BNST')
    m105 = EphysMouse('674-L', 'stress', 'female', 'BNST')
    m106 = EphysMouse('675-R', 'naive', 'female', 'BNST')
    m107 = EphysMouse('674-0', 'stress', 'female', 'PAG')
    m108 = EphysMouse('670-L', 'naive', 'male', 'BNST')
    m109 = EphysMouse('670-RL', 'naive', 'male', 'PAG')
    m110 = EphysMouse('751-R', 'stress', 'female', 'SNL')
    m111 = EphysMouse('756-L', 'naive', 'male', 'SNL')
    m201 = EphysMouse('755-L', 'naive', 'male', 'TS')
    m202 = EphysMouse('755-0', 'naive', 'male', 'TS')
    m203 = EphysMouse('756-0', 'naive', 'female', 'TS')
    m204 = EphysMouse('752-0', 'stress', 'male', 'TS')
    m205 = EphysMouse('752-L', 'stress', 'male', 'TS')
    m206 = EphysMouse('751-RL', 'stress', 'female', 'TS')
    m112 = EphysMouse('751-L', 'stress', 'female', 'SNL')
    m207 = EphysMouse('751-0', 'stress', 'female', 'TS')
    m208 = EphysMouse('757-0', 'naive', 'male', 'TS')
    m113 = EphysMouse('757-L', 'naive', 'male', 'SNL')
    m114 = EphysMouse('758-L', 'naive', 'female', 'SNL')
    m401 = EphysMouse('862-L', 'stress', 'male', 'SNL', True, '40')
    m402 = EphysMouse('855-L', 'naive', 'female', 'SNL', True, '40')
    m403 = EphysMouse('854-L', 'stress', 'female', 'SNL', True, '40')
    m404 = EphysMouse('853-L', 'naive', 'male', 'SNL', True, '40')
    m405 = EphysMouse('862-R', 'stress', 'male', 'SNL', True, '220')
    m406 = EphysMouse('853-R', 'naive', 'male', 'SNL', True, '220')
    m407 = EphysMouse('862-0', 'stress', 'male', 'SNL', True, '220')
    m408 = EphysMouse('853-0', 'naive', 'male', 'SNL', True, '220')
    m409 = EphysMouse('855-RL', 'naive', 'female', 'SNL', True, '220')
    # m501 = EphysMouse('3823-140', 'stress', 'female', 'SNL', True, None, 'XE991')
    # m502 = EphysMouse('1986-127', 'naive', 'female', 'SNL', True, None, 'XE991')
    m602 = EphysMouse('3824-154', 'naive', 'female', 'oIPSC')
    m603 = EphysMouse('7451-157', 'naive', 'male', 'oIPSC')
    m604 = EphysMouse('7451-0', 'naive', 'male', 'oIPSC')
    m605 = EphysMouse('1998-153', 'stress', 'female', 'oIPSC')
    # m701 = EphysMouse('3823-0', 'stress', 'female', 'M-Curr', True, None, 'XE991')
    # m702 = EphysMouse('1986-128', 'naive', 'female', 'M-Curr', True, None, 'XE991')
    m410 = EphysMouse('2006-151', 'naive', 'male', 'PAG')

    # m501.drugSweeps = [range(115, 180),
    #                    range(283, 337),
    #                    range(433, 475),
    #                    range(518, 562)                       
    #                    ]

    # m502.drugSweeps = [range(48, 90),
    #                    range(131, 175),
    #                    range(190, 285),
    #                    range(331, 364)
    #                    ]
    
    # m701.drugSweeps = [range(259, 291),
    #                    range(339, 367)
    #                    ]

    # m702.drugSweeps = [range(163, 198)]

    mouseDict = {
        '610-005': m005,
        '611-006': m006,
        '607-009': m009,
        '607-022': m022,
        '614-023': m023,
        '611-323': m323,
        '608-011': m011,
        '608-021': m021,
        '674-R' : m001,
        '671-L' : m101,
        '675-RL' : m102,
        '675-L' : m103,
        '671-0' : m104,
        '674-L' : m105,
        '675-R' : m106,
        '674-0' : m107,
        '670-L' : m108,
        '670-RL' : m109,
        '751-R' : m110,
        '755-L' : m201,
        '755-0' : m202,
        '756-0' : m203,
        '756-L' : m111,
        '752-0' : m204,
        '752-L' : m205,
        '751-RL' : m206,
        '751-L' : m112,
        '751-0' : m207,
        '757-0' : m208,
        '757-L' : m113,
        '758-L' : m114,
        '862-L' : m401,
        '855-L' : m402,
        '854-L' : m403,
        '853-L' : m404,
        '862-R' : m405,
        '853-R' : m406,
        '862-0' : m407,
        '853-0' : m408,
        '855-RL' : m409,
        # '3823-140' : m501,
        # '1986-127' : m502,
        '3824-154' : m602,
        '7451-157' : m603,
        '7451-0' : m604,
        # '3823-0' : m701,
        # '1986-128' : m702,
        '1998-152' : m605,
        '2006-151' : m410
    }

    return mouseDict


# def findFiringRate(trace,threshold, stimDuration,sampling_rate):
#     spikes, spikeProps = scipy.signal.find_peaks(trace,height=threshold, distance=(0.005*sampling_rate))
#     spikeRate = len(spikes)/float(stimDuration)
#     return spikeRate, spikeProps


protocolDict = {"PA_Ch1_CC Ih -50 pA steps to -150 pA.cfg":
                {"name": "sag CC",
                 "stim": "Current Steps delta -50 pA",
                 "analysisFunc": None},
                "VC test pulse JN.cfg":
                    {"name": "testPulse",
                     "stim": "Test pulse (100 ms)",
                     "analysisFunc": None},
                "PA_Ch1_CC excitability 20 pA steps to 200 pA.cfg":
                    {"name": "I-O curve",
                     "stim": "Current Steps delta 20 pA",
                     "analysisFunc": []},
                "PA_Ch1_CC excitability 5 pA steps to 100 pA.cfg":
                    {"name": "rheobase",
                     "stim": "Current Steps delta 5 pA",
                     "analysisFunc": None},
                "PA_Ch1_CC spont firing for 30s.cfg":
                    {"name": "baselineActivity",
                     "stim": "nothing",
                     "analysisFunc": None},
                "PA_Ch1_WC_VC_test pulse.cfg":
                    {"name": "testPulse",
                     "stim": "Test pulse (50 ms)",
                     "analysisFunc": None},
                "PA_Ch1_WC_VC_Ih test.cfg":
                    {"name": "VC Ih",
                     "stim": None,
                     "analysisFunc": None
                     },
                "JN_VC_green-light_1ms.cfg":
                    {"name": "Opto single pulse 1ms",
                     "stim": None,
                     "analysisFunc": None
                     },
                "JN_ES_MZ_VC_PPR 1ms Green light 100ms ISI.cfg":
                    {"name": "Opto PPR 1ms 100ms ISI",
                     "stim": None,
                     "analysisFunc": None
                     },
                "JN_ES_MZ_VC_spont currents for 31s.cfg":
                    {"name": "Spont IPSCs 31s",
                     "stim": None,
                     "analysisFunc": None
                     },
                "JN_Ch1_VC_mCurrent_hyperpolStep.cfg":
                    {"name": "M-current hyperpol step",
                     "stim": None,
                     "analysisFunc": None
                     },

                }

#! ACUS AUGUST 2023 MOUSE INFO LIST
# mouseDict = {"610-005" : {'condition' : 'naive',
#                           'sex' : 'female',
#                           'holdingSweeps' : [],
#                           'use' : False
#                         },
#              "611-006" :  {'condition' : 'stress',
#                           'sex' : 'male',
#                           'holdingSweeps' : [],
#                           'use' : True
#                           },
#              "607-009" : {'condition' : 'stress',
#                           'sex' : 'male',
#                           'holdingSweeps' : [],
#                           'use' : True
#                           },
#              "607-022" : {'condition' : 'stress',
#                           'sex' : 'male',
#                           'holdingSweeps' : [range(1027, 1038), range(1112, 1153)],
#                           'use' : True
#                           },
#              "614-023" : {'condition' : 'naive',
#                           'sex' : 'female',
#                           'holdingSweeps' : [],
#                           'use' : True
#                           },
#              "611-323" : {'condition' : 'stress',
#                           'sex' : 'male',
#                           'holdingSweeps' : [],
#                           'use' : True
#                           },
#              "608-011" : {'condition' : 'naive',
#                           'sex' : 'male',
#                           'holdingSweeps' : [],
#                           'use' : True
#                           },
#              "608-021" : {'condition' : 'naive',
#                           'sex' : 'male',
#                           'holdingSweeps' : [],
#                           'use' : True
#                           }

#             }

# %%