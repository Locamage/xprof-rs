use crate::counter_ids::{V6E_IDS, V7X_IDS};
use rustc_hash::FxHashMap;

const GIB: f64 = 1024.0 * 1024.0 * 1024.0;
const INSTRUCTIONS: &str = "instructions";
const CYCLES: &str = "cycles";
const BYTES: &str = "bytes";
const PERCENT: &str = "percent";
const VECTOR_ALUS_PER_CORE: u64 = 4;
const MXUS_PER_TENSOR_CORE: u64 = 2;
const SC_CORES_PER_DIE: usize = 2;
const SC_TILES: usize = 16;
const HBM_BYTES_PER_BEAT: u64 = 32;
const ICI_BYTES_PER_FLIT: u64 = 128;
const PEAK_ICI_BYTES_PER_SECOND: f64 = 294947368421.0526;
const ICI_FREQUENCY_HZ: f64 = 1e9;
const TC_V6E: &str = "VF_CHIP_TC_TCS_TC_MISC_TCS_STATS_TCS_STATS_COUNTERS_UNPRIVILEGED_COUNT_";
const HBM_SUFFIX: &str = "_CMN_HI_FREQ_STATS_COUNTERS_UNPRIVILEGED_";
const SC_SUFFIX: &str = "_SC_STATS_COUNTERS_UNPRIVILEGED_COUNT_";
const POWER_STATS: &str = "PWRMGR_PWRMGR_TC_THROTTLE_CORE_DEBUG_STATS_UNPRIVILEGED_";
const SKIPPED_CLOCKS: [(&str, &str); 3] =
    [("Clocks Skipped", "CLOCKS_SKIPPED"), ("Ext Throttle Clocks Skipped", "EXT_THROTTLE_CLOCKS_SKIPPED"), ("LDIDT Droop Clocks Skipped", "LDIDT_DROOP_CLOCKS_SKIPPED")];
const TEC_SLOTS: [(&str, &str); 4] = [
    ("TEC Scalar Issue", "TEC_SCALAR_ISSUE"),
    ("TEC Scalar Issue: S0 slot", "TEC_S0_INSTRUCTION"),
    ("TEC Scalar Issue: S1 slot", "TEC_S1_INSTRUCTION"),
    ("TEC Scalar Issue: Smisc slot", "TEC_SMISC_INSTRUCTION"),
];
const SCS_SLOTS: [(&str, &str); 4] = [("SCS Scalar Issue", "SCALAR_ISSUE"), ("SCS: S0 Slot", "S0_INSTRUCTION"), ("SCS: S1 Slot", "S1_INSTRUCTION"), ("SCS: Smisc Slot", "SMISC_INSTRUCTION")];
const V6E_VECTOR_SLOTS: [(&str, &str); 8] = [
    ("SC Vector Issue", "VECTOR_ISSUE"),
    ("SC Vector Issue:V0 Slot", "V0_INSTRUCTION"),
    ("SC Vector Issue: V1 Slot", "V1_INSTRUCTION"),
    ("SC Vector Issue: V2 Slot", "V2_INSTRUCTION"),
    ("SC Vector Issue: VLD Slot", "VLD_INSTRUCTION"),
    ("SC Vector Issue: VST Slot", "VST_INSTRUCTION"),
    ("SC Vector Issue:VEX Slot", "VEX_INSTRUCTION"),
    ("SC Vector Issue:VRES Slot", "VRES_INSTRUCTION"),
];
const V7X_VECTOR_SLOTS: [(&str, &str); 8] = [
    ("SC Vector Issue", "VECTOR_ISSUE"),
    ("SC Vector Issue: V0 Slot", "V0_INSTRUCTION"),
    ("SC Vector Issue: V1 Slot", "V1_INSTRUCTION"),
    ("SC Vector Issue: V2 Slot", "V2_INSTRUCTION"),
    ("SC Vector Issue: VLD Slot", "VLD_INSTRUCTION"),
    ("SC Vector Issue: VST Slot", "VST_INSTRUCTION"),
    ("SC Vector Issue: VEX Slot", "VEX_INSTRUCTION"),
    ("SC Vector Issue: VRES Slot", "VRES_INSTRUCTION"),
];
const VPU_OPS: [&str; 5] = ["FADD", "FMUL", "MISC", "INT", "FMAC"];
const XLU_INSTRUCTIONS: [&str; 4] = ["PACKED_XLU", "ROTATE_PERMUTE_INSTRUCTION_XLU", "ROTATE_PERMUTE_SET_INSTRUCTION_XLU", "TRANSPOSE_XLU"];
const V6E_PRECISIONS: [(&str, &[(u64, &str)]); 3] =
    [("MXU BF16", &[(4, "LMR_BF16"), (8, "VREG_F8"), (8, "VREG_BF16"), (4, "VREG_F32")]), ("MXU I8", &[(2, "LMR_I8"), (8, "VREG_I8")]), ("MXU I4", &[(2, "LMR_I4"), (8, "VREG_I4")])];
const V7X_PRECISIONS: [(&str, &[(u64, &str)]); 2] = [("MXU BF16", &[(4, "LMR_BF16"), (8, "VREG_BF16"), (4, "VREG_F32")]), ("MXU E4M3 + E5M2", &[(8, "VREG_F8")])];

pub type Counters = FxHashMap<u64, u64>;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Device {
    V6e,
    V7x,
}

#[derive(Debug, PartialEq)]
pub struct Metric {
    pub node: u64,
    pub name: String,
    pub achieved: f64,
    pub peak: f64,
    pub unit: &'static str,
}

impl Device {
    pub fn from_type(device_type: &str) -> Option<Self> {
        match device_type {
            "TPU v7x" => Some(Self::V7x),
            "TPU v6 Lite" => Some(Self::V6e),
            _ => None,
        }
    }

    pub fn frequency_hz(self) -> f64 {
        if self == Self::V6e { 1.75e9 } else { 1.9e9 }
    }

    fn id(self, name: &str) -> u64 {
        if self == Self::V6e { V6E_IDS[name] } else { V7X_IDS[name] }
    }

    fn tensor_counter(self, core: usize, name: &str) -> u64 {
        match self {
            Self::V6e => self.id(&format!("{TC_V6E}{name}")),
            Self::V7x => self.id(&format!("VF_CHIP_DIE{}_TC_TCS_TC_MISC_TCS_STATS_TCS_STATS_COUNTERS_UNPRIVILEGED_COUNT_{name}", die(core))),
        }
    }

    fn power_counter(self, core: usize, name: &str) -> u64 {
        self.id(&format!("VF_CHIP_DIE{}_{POWER_STATS}{name}", die(core)))
    }

    pub fn cycle_counters(self) -> Vec<u64> {
        match self {
            Self::V6e => vec![self.tensor_counter(0, "CYCLES")],
            Self::V7x => (0..2).map(|core| self.power_counter(core, "CYCLE_COUNT")).collect(),
        }
    }
}

pub(crate) fn count(counters: &Counters, id: u64) -> u64 {
    counters.get(&id).copied().unwrap_or(0)
}

fn total(counters: &Counters, ids: impl IntoIterator<Item = u64>) -> u64 {
    ids.into_iter().fold(0u64, |sum, id| sum.wrapping_add(count(counters, id)))
}

fn die(core: usize) -> usize {
    usize::from(core != 0)
}

fn add(metrics: &mut Vec<Metric>, node: usize, name: &str, achieved: f64, peak: f64, unit: &'static str) {
    metrics.push(Metric { node: node as u64, name: name.to_string(), achieved, peak, unit });
}

pub fn compute(counters: &Counters, device: Device) -> Vec<Metric> {
    let mut metrics = Vec::new();
    let cores = if device == Device::V6e { 1 } else { 2 };
    for core in 0..cores {
        tensor_core(counters, device, core, &mut metrics);
        bandwidth(counters, device, core, &mut metrics);
        ici(counters, device, &mut metrics);
    }
    for (die, core) in (0..cores).flat_map(|die| (0..SC_CORES_PER_DIE).map(move |core| (die, core))) {
        sparse_core(counters, device, die, core, &mut metrics);
    }
    metrics
}

fn tensor_core(counters: &Counters, device: Device, core: usize, metrics: &mut Vec<Metric>) {
    let counter = |name: &str| count(counters, device.tensor_counter(core, name));
    let total = |names: &[String]| names.iter().fold(0u64, |sum, name| sum.wrapping_add(counter(name)));
    let power = |name: &str| count(counters, device.power_counter(core, name));
    let cycles = if device == Device::V6e { counter("CYCLES") } else { power("CYCLE_COUNT") };
    if cycles == 0 {
        return;
    }
    let cycles_f = cycles as f64;
    if device == Device::V7x {
        for (name, counter_name) in SKIPPED_CLOCKS {
            add(metrics, core, name, power(counter_name) as f64, cycles_f, CYCLES);
        }
    }
    let indexed = |prefix: &str, count: usize| (0..count).map(|index| format!("{prefix}_{index}")).collect::<Vec<_>>();
    add(metrics, core, "Scalar Unit", total(&indexed("SCALAR_ALU_INSTRUCTION", 2)) as f64, cycles.wrapping_mul(2) as f64, INSTRUCTIONS);
    let vector_peak = cycles.wrapping_mul(VECTOR_ALUS_PER_CORE) as f64;
    add(metrics, core, "Vector ALUs", total(&indexed("VECTOR_ALU_INSTRUCTION", 4)) as f64, vector_peak, INSTRUCTIONS);
    let vpu_ops: Vec<String> = VPU_OPS.iter().flat_map(|op| indexed(&format!("VPU_VALU_{op}_OPS"), 4)).collect();
    add(metrics, core, "VPU Utilization", total(&vpu_ops) as f64, vector_peak, INSTRUCTIONS);
    add(metrics, core, "Vmem Stores", counter("VST_INSTRUCTION") as f64, cycles_f, INSTRUCTIONS);
    add(metrics, core, "Vmem Loads", total(&indexed("VLD_INSTRUCTION", 2)) as f64, cycles.wrapping_mul(2) as f64, INSTRUCTIONS);
    let busy = |metrics: &mut Vec<Metric>, values: [u64; 3], names: [&str; 4]| {
        for (name, value) in names.into_iter().zip(values) {
            add(metrics, core, name, value as f64, cycles_f, CYCLES);
        }
        add(metrics, core, names[3], 0.5 * values[1] as f64 + values[2] as f64, cycles_f, CYCLES);
    };
    busy(metrics, ["MXU_BUSY_0", "MXU_BUSY_1", "MXU_BUSY_2"].map(counter), ["No MXU Busy", "1 MXU Busy", "2 MXU Busy", "Avg MXU Busy"]);
    let mxu_peak = cycles.wrapping_mul(MXUS_PER_TENSOR_CORE) as f64;
    let precisions: &[(&str, &[(u64, &str)])] = if device == Device::V6e { &V6E_PRECISIONS } else { &V7X_PRECISIONS };
    let weighted = |unit: usize, terms: &[(u64, &str)]| terms.iter().fold(0u64, |sum, &(weight, name)| sum.wrapping_add(weight.wrapping_mul(counter(&format!("MATMUL_{name}_MXU_{unit}")))));
    for unit in 0..2 {
        add(metrics, core, &format!("MXU{unit}"), precisions.iter().fold(0u64, |sum, (_, terms)| sum.wrapping_add(weighted(unit, terms))) as f64, cycles_f, CYCLES);
    }
    for (name, terms) in precisions {
        add(metrics, core, name, weighted(0, terms).wrapping_add(weighted(1, terms)) as f64, mxu_peak, CYCLES);
    }
    add(metrics, core, "MXU matpush", total(&indexed("MATPUSH_CYCLES_MXU", 2)) as f64, mxu_peak, CYCLES);
    busy(metrics, ["XLU_BUSY_0", "XLU_BUSY_1", "XLU_BUSY_2"].map(counter), ["No XLU Busy", "1 XLU Busy", "2 XLUs Busy", "Avg XLU Busy"]);
    let xlu_peak = (cycles / if device == Device::V6e { 4 } else { 1 }) as f64;
    for unit in 0..2 {
        add(metrics, core, &format!("XLU{unit}"), total(&XLU_INSTRUCTIONS.map(|name| format!("{name}_{unit}"))) as f64, xlu_peak, INSTRUCTIONS);
    }
}

fn sparse_core(counters: &Counters, device: Device, die_index: usize, core: usize, metrics: &mut Vec<Metric>) {
    let (sc, node) = (die(core), if device == Device::V6e { core } else { (die_index << 1) + core });
    let prefix = if device == Device::V6e { format!("VF_CHIP_SC_{sc}") } else { format!("VF_CHIP_DIE{}_SC_{sc}", die(die_index)) };
    let tiles = |subsystem: &str, name: &str| total(counters, (0..SC_TILES).map(|tile| device.id(&format!("{prefix}_{subsystem}_{tile}{SC_SUFFIX}{name}"))));
    let (scalar, vector, slots) = if device == Device::V6e { ("SCT", "SCT", &V6E_VECTOR_SLOTS) } else { ("SCTD", "SCTC", &V7X_VECTOR_SLOTS) };
    let cycle_peak = count(counters, device.id(&format!("{prefix}_{scalar}_0{SC_SUFFIX}CYCLES"))).wrapping_mul(SC_TILES as u64) as f64;
    for (metric, name) in TEC_SLOTS {
        add(metrics, node, metric, tiles(scalar, name) as f64, cycle_peak, INSTRUCTIONS);
    }
    for (metric, name) in slots {
        add(metrics, node, metric, tiles(vector, name) as f64, cycle_peak, INSTRUCTIONS);
    }
    let sequencer = |name: &str| count(counters, device.id(&format!("{prefix}_SCS{SC_SUFFIX}{name}"))) as f64;
    for (metric, name) in SCS_SLOTS {
        add(metrics, node, metric, sequencer(name), sequencer("CYCLES"), INSTRUCTIONS);
    }
}

fn hbm_counters(device: Device, core: usize, read: bool) -> Vec<u64> {
    let kinds: &[&str] = if read { &["RD_RESP"] } else { &["WR_REQ", "PARTIAL_WRITE_REQ"] };
    let (stacks, prefix) = if device == Device::V6e { (2, "VF_CHIP_HBM".to_string()) } else { (4, format!("VF_CHIP_DIE{}_HBM", die(core))) };
    let controller = if device == Device::V6e { "HBMC" } else { "SS_HBMC" };
    let mut ids = Vec::new();
    for stack in 0..stacks {
        for index in 0..16 {
            for channel in 0..2 {
                ids.extend(kinds.iter().map(|kind| device.id(&format!("{prefix}_{stack}_{controller}_{index}{HBM_SUFFIX}{kind}_PS{channel}"))));
            }
        }
    }
    ids
}

fn bandwidth(counters: &Counters, device: Device, core: usize, metrics: &mut Vec<Metric>) {
    let read = total(counters, hbm_counters(device, core, true)).wrapping_mul(HBM_BYTES_PER_BEAT);
    let written = total(counters, hbm_counters(device, core, false)).wrapping_mul(HBM_BYTES_PER_BEAT);
    let cycles = count(counters, device.tensor_counter(core, "CYCLES"));
    if cycles == 0 {
        return;
    }
    let peak_bandwidth = if device == Device::V6e { 1525.5 * GIB } else { 3433.0 * GIB };
    let total = read.wrapping_add(written) as f64;
    add(metrics, 0, &format!("HBM Rd+Wr - core {core}"), total, peak_bandwidth * (cycles as f64 / device.frequency_hz()), BYTES);
    if total > 0.0 {
        add(metrics, core, "HBM Read Ratio", read as f64, total, PERCENT);
        add(metrics, core, "HBM Write Ratio", written as f64, total, PERCENT);
    }
}

fn ici(counters: &Counters, device: Device, metrics: &mut Vec<Metric>) {
    let flits = |kind: &str| match device {
        Device::V6e => total(counters, (0..2).map(|index| device.id(&format!("VF_CHIP_OCI_ICR_PERF_COUNTERS_FIFO_STATS_COUNTERS_UNPRIVILEGED_BN_{kind}_REQ_{index}_INSERTION_COUNT")))),
        Device::V7x => total(counters, (0..4).map(|index| device.id(&format!("VF_CHIP_DIE1_OCI_ICR_{index}_PERF_COUNTERS_FIFO_STATS_COUNTERS_UNPRIVILEGED_BN_{kind}_REQ_0_INSERTION_COUNT")))),
    };
    let (read, written) = (flits("RD").wrapping_mul(ICI_BYTES_PER_FLIT), flits("WR").wrapping_mul(ICI_BYTES_PER_FLIT));
    let cycles = count(counters, device.cycle_counters()[0]);
    if cycles == 0 {
        return;
    }
    let peak = PEAK_ICI_BYTES_PER_SECOND * (cycles as f64 / ICI_FREQUENCY_HZ);
    add(metrics, 0, "ICI (Read)", read as f64, peak, BYTES);
    add(metrics, 0, "ICI (Write)", written as f64, peak, BYTES);
}

#[cfg(test)]
#[path = "tests/inline/utilization.rs"]
mod tests;
