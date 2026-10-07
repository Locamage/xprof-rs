use crate::hlo::proto_text::json_string;
use crate::tools::event_fractions::{Fractions, accumulate, analyze};
use crate::tools::input_pipeline_analyzer::fixed;
use crate::xplane::steps::SPARSE_CORE_START;
use std::collections::BTreeMap;
use std::path::PathBuf;

const HBM_LOW: f64 = 50.0;
const HBM_HIGH: f64 = 70.0;
const MXU_LOW: f64 = 50.0;
const MXU_HIGH: f64 = 70.0;
const INFEED_PERCENT: f64 = 10.0;
const COLLECTIVE_PERCENT: f64 = 30.0;
const DATA_SHUFFLE_PERCENT: f64 = 30.0;
const SPARSE_CORE_PERCENT: f64 = 10.0;
const DATA_TRANSFER_PERCENT: f64 = 30.0;
const HOST_PROCESSING_PERCENT: f64 = 50.0;
const TENSOR_CORE_IDLE_PERCENT: f64 = 10.0;
const SPECIAL_OP_PERCENT: f64 = 10.0;
const DEBUG_PRINT_PERCENT: f64 = 5.0;
const ASYNC_DONE_PERCENT: f64 = 10.0;
const MEMORY_HIGH: f64 = 50.0;
const ZERO_EPSILON: f64 = 1e-6;
const ANALYZED_EVENTS: [&str; 2] = ["barrier-cores", "debug_print"];
const SPECIAL_OP: &str = "barrier-cores";
const DEBUG_PRINT: &str = "debug_print";
const STRAGGLER_THRESHOLD: f64 = 3.5;
const MAD_THRESHOLD: f64 = 0.1;
const MODIFIED_Z: f64 = 0.6745;
const THIRD_PARTY_RULES: [&str; 1] = ["BarrierCoresRule"];
const COLLECTIVES: [&str; 17] = [
    "all-reduce", "all-reduce fusion", "all-reduce-scatter fusion", "all-to-all", "all-gather", "all-gather-start", "all-gather-done", "all-gather fusion", "reduce-scatter", "collective-permute",
    "collective-permute-done", "collective-permute-start", "megacore fusion", "host recv", "host recv-done", "host send", "host send-done",
];
const DATA_SHUFFLE: [&str; 14] =
    ["broadcast", "concatenate", "data formatting", "dynamic-slice", "dynamic-update-slice", "gather", "pad", "reverse", "scatter", "select", "select-and-scatter", "shuffle", "slice", "sort"];
const ASYNC_DONE_PREFIXES: [&str; 5] = ["all-gather", "all-reduce", "all-to-all", "ragged-all-to-allreduce-gather", "reduce-scatter"];

pub(crate) type Data<T> = Result<T, String>;
type Results = BTreeMap<String, Fractions>;

#[derive(Default, Clone)]
pub struct Overview {
    pub hbm_percent: f64,
    pub mxu_percent: f64,
}

#[derive(Default, Clone)]
pub struct InputPipeline {
    pub input_percent: f64,
    pub enqueue_us: f64,
    pub demanded_file_read_us: f64,
    pub advanced_file_read_us: f64,
    pub preprocessing_us: f64,
    pub unclassified_non_enqueue_us: f64,
    pub tpu: Option<(f64, f64)>,
    pub step_time_ms: f64,
}

#[derive(Default, Clone)]
pub struct Step {
    pub cores: Vec<(u32, u64)>,
    pub categories: Vec<(String, u64)>,
}

#[derive(Default, Clone)]
pub struct Node {
    pub name: String,
    pub raw_time: f64,
    pub children: Vec<Self>,
}

pub trait ToolData {
    fn overview(&self) -> Data<Overview> {
        Err("overview is not consulted by the registered rules".into())
    }
    fn input_pipeline(&self) -> Data<InputPipeline> {
        Err("input pipeline is not consulted by the registered rules".into())
    }
    fn event_fractions(&self, event: &str) -> Data<Fractions>;
    fn steps(&self) -> Data<Vec<Step>> {
        Err("op stats are not consulted by the registered rules".into())
    }
    fn op_profile(&self) -> Data<Node> {
        Err("op profile is not consulted by the registered rules".into())
    }
    fn memory(&self) -> Data<Vec<(f64, f64)>> {
        Err("memory profile is not consulted by the registered rules".into())
    }
}

pub struct Suggestion {
    pub rule: &'static str,
    pub text: String,
}

pub(crate) struct Rule {
    pub(crate) name: &'static str,
    pub(crate) meets: fn(&dyn ToolData) -> bool,
    pub(crate) generate: fn(&dyn ToolData) -> Data<String>,
}

struct Straggler {
    hostname: String,
    percent: f64,
}

fn stat_average(values: impl Iterator<Item = f32>) -> Option<f64> {
    let (sum, count) = values.fold((0.0f64, 0u64), |(sum, count), value| (sum + f64::from(value), count + 1));
    (count > 0).then(|| f64::from((sum / count as f64) as f32))
}

fn median(mut values: Vec<f64>) -> f64 {
    let middle = values.len() / 2;
    values.select_nth_unstable_by(middle, f64::total_cmp);
    values[middle]
}

fn hbm(data: &dyn ToolData) -> Data<f64> {
    data.overview().map(|overview| overview.hbm_percent)
}

fn mxu(data: &dyn ToolData) -> Data<f64> {
    data.overview().map(|overview| overview.mxu_percent)
}

fn input_percent(data: &dyn ToolData) -> Data<f64> {
    data.input_pipeline().map(|input| input.input_percent)
}

fn percent_of_input(data: &dyn ToolData, enqueue: bool) -> Data<f64> {
    let input = data.input_pipeline()?;
    let other = input.demanded_file_read_us + input.advanced_file_read_us + input.preprocessing_us + input.unclassified_non_enqueue_us;
    let total = input.enqueue_us + other;
    Ok(if total == 0.0 { 0.0 } else { (if enqueue { input.enqueue_us } else { other }) / total * 100.0 })
}

fn step_fractions(data: &dyn ToolData, categories: &[&str]) -> Data<Vec<f32>> {
    let steps = data.steps()?;
    let fractions = steps.iter().filter(|step| !step.cores.is_empty()).filter_map(|step| {
        let total: u64 = step.cores.iter().filter(|(core, _)| *core < SPARSE_CORE_START).map(|(_, duration)| duration).sum();
        let matched: u64 = step.categories.iter().filter(|(category, _)| categories.contains(&category.as_str())).map(|(_, time)| time).sum();
        (total != 0).then(|| matched as f32 / total as f32)
    });
    Ok(fractions.collect())
}

fn average_percent(fractions: &[f32]) -> f64 {
    stat_average(fractions.iter().copied()).map_or(0.0, |average| average * 100.0)
}

pub(crate) fn collective_percent(data: &dyn ToolData) -> Data<f64> {
    step_fractions(data, &COLLECTIVES).map(|fractions| average_percent(&fractions))
}

fn data_shuffle_percent(data: &dyn ToolData) -> Data<f64> {
    step_fractions(data, &DATA_SHUFFLE).map(|fractions| average_percent(&fractions))
}

fn tpu_percent(data: &dyn ToolData, sparse_core: bool) -> Data<f64> {
    let input = data.input_pipeline()?;
    let (idle, sparse) = input.tpu.ok_or_else(|| "Failed to unpack TpuStepTimeBreakdown.".to_string())?;
    Ok(if input.step_time_ms == 0.0 { 0.0 } else { (if sparse_core { sparse } else { idle }) / input.step_time_ms * 100.0 })
}

fn event_percent(data: &dyn ToolData, event: &str) -> Data<f64> {
    let fractions = data.event_fractions(event)?;
    Ok(stat_average(fractions.chips.values().flatten().copied()).map_or(0.0, |average| average * 100.0))
}

fn host_event_percents(data: &dyn ToolData, event: &str) -> Data<Vec<(String, f64)>> {
    let fractions = data.event_fractions(event)?;
    Ok(fractions.hosts.iter().filter_map(|(host, values)| Some((host.clone(), stat_average(values.iter().copied())? * 100.0))).collect())
}

fn stragglers(data: &dyn ToolData, event: &str) -> Data<Vec<Straggler>> {
    let hosts = host_event_percents(data, event)?;
    if hosts.len() < 3 {
        return Ok(Vec::new());
    }
    let center = median(hosts.iter().map(|(_, percent)| *percent).collect());
    let mad = median(hosts.iter().map(|(_, percent)| (percent - center).abs()).collect());
    Ok(hosts
        .into_iter()
        .filter_map(|(hostname, percent)| {
            let score = if mad != 0.0 {
                MODIFIED_Z * (percent - center) / mad.max(MAD_THRESHOLD)
            } else if (percent - center).abs() > ZERO_EPSILON {
                f64::INFINITY
            } else {
                0.0
            };
            (score.abs() > STRAGGLER_THRESHOLD).then_some(Straggler { hostname, percent })
        })
        .collect())
}

fn async_done_percent(data: &dyn ToolData) -> Data<f64> {
    let root = data.op_profile()?;
    let Some(first) = root.children.first() else { return Ok(0.0) };
    let time: f64 = first
        .children
        .iter()
        .filter(|node| node.name == "async-done")
        .flat_map(|node| &node.children)
        .filter(|child| ASYNC_DONE_PREFIXES.contains(&child.name.split('.').next().unwrap_or("")))
        .map(|child| child.raw_time)
        .fold(0.0, |total, time| total + time);
    Ok(time / root.raw_time * 100.0)
}

fn peak_memory_percent(data: &dyn ToolData) -> Data<f64> {
    let (usage, capacity) = data.memory()?.iter().fold((0.0, 0.0), |(usage, capacity), (peak, total)| (usage + peak, capacity + total));
    Ok(usage / capacity * 100.0)
}

fn latency_bound(data: &dyn ToolData) -> bool {
    matches!((mxu(data), hbm(data)), (Ok(mxu), Ok(hbm)) if mxu < MXU_LOW && hbm < HBM_LOW)
}

fn input_bound(data: &dyn ToolData) -> bool {
    input_percent(data).is_ok_and(|percent| percent > INFEED_PERCENT)
}

fn one(value: f64) -> String {
    fixed(value, 1)
}

fn barrier_cores(data: &dyn ToolData) -> Data<String> {
    let percent = event_percent(data, SPECIAL_OP)?;
    let balance = match stragglers(data, SPECIAL_OP) {
        Ok(found) if !found.is_empty() => {
            let list: String = found.iter().map(|straggler| format!("<li>Host <b>{}</b> average barrier-cores time fraction: <b>{}%</b></li>", straggler.hostname, one(straggler.percent))).collect();
            format!("<li><b>Investigate Stragglers:</b> The following hosts are identified as potential stragglers with significantly different barrier time fraction compared to others:<ul>{list}</ul></li>")
        }
        _ => "<li><b>Investigate Workload Balance:</b> Check for stragglers, i.e., workers that are significantly slower than others. Uneven workloads can cause faster workers to wait at the barrier.</li>".into(),
    };
    Ok(format!(
        "<p>Your program is likely bottlenecked by <b>TPU {SPECIAL_OP}</b> operations: <b> an average of {}% of each step time</b> is spent on these operations. This often indicates a synchronization issue between workers in a distributed training setup. Please consider the following optimizations:</p><ul>{balance}<li><b>Optimize Collective Operations:</b> Operations like AllReduce involve synchronization. Ensure they are used efficiently. Check the size of data being communicated.</li><li><b>Check Network:</b> Network latency or bandwidth can be a bottleneck for distributed operations, causing workers to wait longer at barriers.</li><li><b>Improve Data Input Pipeline:</b> Ensure your data loading and preprocessing pipeline is efficient and balanced across all workers. A slow input pipeline on one worker can stall all others.</li></ul>",
        one(percent)
    ))
}

fn collective_bound(data: &dyn ToolData) -> Data<String> {
    Ok(format!(
        "<p>Your program is likely bottlenecked by <b>Collective operations (e.g., AllReduce, AllGather)</b>. An average of <b>{}%</b> of each step is spent on these operations. This suggests that your model is collective communication-bound. Please consider the following optimizations:</p><ul><li><b>Overlap Communication with Computation:</b> Try to schedule collective operations to overlap with computation to hide communication latency.</li><li><b>Investigate Workload Balance and Input Pipeline:</b> Check for stragglers, i.e., workers that are significantly slower than others. Uneven workloads and data loading can cause faster workers to wait during collective operations.</li><li><b>Optimize Collective Operations:</b> Reduce the amount of data transferred between devices by using lower precision data formats for gradients (e.g., bfloat16).</li><li><b>Check Network:</b> Network latency or bandwidth can be a bottleneck for distributed operations, causing workers to wait longer during collectives.</li></ul>",
        one(collective_percent(data)?)
    ))
}

fn compute_bound(data: &dyn ToolData) -> Data<String> {
    let (hbm, mxu) = (hbm(data)?, mxu(data)?);
    Ok(format!(
        "<p>Your program is likely bottlenecked by <b>Compute Operations</b>: High MXU utilization of <b>{}%</b> and low HBM Bandwidth utilization of <b>{}%</b> indicates that the primary bottleneck is the raw processing power of the hardware. Please consider the following optimizations: </p><ul><li><b>Use Mixed Precision:</b> Using bfloat16 for computations and storing weights can significantly speed up matrix multiplications and reduce memory usage. Ensure that this does not negatively impact your model's convergence.</li><li><b>Optimize Your Kernels:</b> If you are using custom operations, profile them to identify any inefficiencies. For standard operations, ensure you are using the latest version of your framework and libraries (e.g., CUDA, cuDNN for GPUs) which often include optimized kernels.</li><li><b>Experiment with Batch Size:</b> While a large batch size can improve hardware utilization, an excessively large batch size might not always be optimal. Experiment with different batch sizes to find the sweet spot for your specific model and hardware.</li><li><b>Switching to a Mixture of Experts (MoE) Strategy:</b> Consider switching to a Mixture of Experts (MoE) architecture by replacing one massive FFN with several smaller expert networks to leverage SparseCores. The SparseCore handles the gating logic—deciding which specific expert should handle which token. It performs the routing (scatter/gather). It effectively trades excess Compute (MXU) load for Memory Bandwidth and Sparse Core utilization.</li></ul>",
        one(mxu),
        one(hbm)
    ))
}

fn data_shuffle_bound(data: &dyn ToolData) -> Data<String> {
    Ok(format!(
        "<p>Your program is likely bottlenecked by <b>Data Shuffle operations (e.g., sort, gather, scatter)</b>. An average of <b>{}%</b> of each step is spent on these operations. Please consider the following optimizations:</p><ul><li><b>Optimize Gather, Scatter Operations:</b> If gather, scatter operations are the bottleneck, review their implementation. Ensure that the dimensions of tensors involved in gather and scatter operations are multiples of 128 (common for most TPUs).</li></ul>",
        one(data_shuffle_percent(data)?)
    ))
}

fn data_transfer_bound(data: &dyn ToolData) -> Data<String> {
    let (input, enqueue) = (input_percent(data)?, percent_of_input(data, true)?);
    Ok(format!(
        "<p>Your program is likely bottlenecked by <b>data transfer</b> between Host and Device: <b>{}% of the total step time</b> is spent on enqueuing data to the device. Please consider the following optimizations:</p><ul><li><b>Combine small data chunks:</b> Transferring many small chunks of data can be inefficient. Try to batch them into fewer, larger transfers.</li><li><b>Check transfer size:</b> Ensure the size of data being transferred in each batch is optimal for the hardware.</li><li><b>Use prefetching:</b> Overlap data transfer with computation.</li></ul>",
        one(input * enqueue / 100.0)
    ))
}

fn debug_print(data: &dyn ToolData) -> Data<String> {
    let high: Vec<(String, f64)> = host_event_percents(data, DEBUG_PRINT).unwrap_or_default().into_iter().filter(|(_, percent)| *percent >= DEBUG_PRINT_PERCENT).collect();
    let max_percent = high.iter().fold(0.0f64, |max, (_, percent)| if *percent > max { *percent } else { max });
    let list: String = high.iter().map(|(host, percent)| format!("<li>Host <b>{host}</b> average debug_print time fraction: <b>{}%</b></li>", one(*percent))).collect();
    Ok(format!(
        "<p>Your program is likely bottlenecked by <b>{DEBUG_PRINT}</b> operations: <b> up to {}% of step time</b> is spent on these operations on some hosts. This often indicates the debug print operations causes busy waiting for other workers in a distributed training setup. Please consider the following optimizations:</p><ul><li><b>Investigate Hosts with High Debug Print Time:</b> The following hosts have high debug_print time fraction compared to others:<ul>{list}</ul></li><li><b>Check for debug print usage:</b> Check for <code>jax.debug.print</code> calls in your code.</li></ul>",
        one(max_percent)
    ))
}

fn host_processing_bound(data: &dyn ToolData) -> Data<String> {
    let (input, other) = (input_percent(data)?, percent_of_input(data, false)?);
    Ok(format!(
        "<p>Your program is likely bottlenecked by <b>Host-side Processing</b> in the input pipeline: <b>{}% of the total step time</b> is spent on host-side input data processing. Please consider the following optimizations:</p><ul><li><b>Optimize Data Reading:</b> Ensure efficient file reading patterns. Use prefetching and interleaving to load data in parallel and in advance.</li><li><b>Parallelize Data Preprocessing:</b> Utilize parallel processing techniques for CPU-bound preprocessing steps.</li><li><b>Offline Preprocessing:</b> For static datasets, consider performing expensive preprocessing steps offline and saving the results.</li><li><b>Tuning Parameters:</b> Experiment with buffer sizes, the number of parallel threads, and prefetch distances in your input pipeline to find the best settings.</li></ul>",
        one(input * other / 100.0)
    ))
}

fn input_bound_text(data: &dyn ToolData) -> Data<String> {
    Ok(format!(
        "<p>Your program is likely <b>input-bound</b> : <b>{}% of step time</b> is spent on input operations. Please consider the following documentations to optimize your input pipeline:</p><ul></ul>",
        one(input_percent(data)?)
    ))
}

fn memory_bound(data: &dyn ToolData) -> Data<String> {
    let (hbm, mxu) = (hbm(data)?, mxu(data)?);
    Ok(format!(
        "<p>Your program is likely bottlenecked by <b>Memory Operations</b>: High HBM Bandwidth utilization of <b>{}%</b> and low MXU utilization of <b>{}%</b> indicates that processors are often waiting for data. Please consider the following optimizations:</p><ul><li><b>Increase the Batch Size:</b> A larger batch size increases the computational work (the matrix multiplications in your model) done for each data loading step. This improves the ratio of computation to memory access, which should directly increase MXU utilization.</li><li><b>Utilize Gradient Accumulation:</b> This technique processes several smaller batches sequentially and only updates the model weights after accumulating the gradients from all of them. This simulates a larger effective batch size without a proportional increase in memory usage.</li><li><b>Increase Model Depth or Width:</b> Make the model larger by adding more layers or increasing the size of the hidden dimensions to improve the MXU utilization.</li><li><b>Optimize the Data Format:</b> Store your dataset in a format optimized for high-throughput reads, such as TFRecord for TensorFlow or Petastorm for PyTorch. These formats are more efficient than reading individual image or text files.</li></ul>",
        one(hbm),
        one(mxu)
    ))
}

fn tensor_core_idle_bound(data: &dyn ToolData) -> Data<String> {
    Ok(format!(
        "<p>Your program is likely bottlenecked by <b>TensorCore Idle Time</b>: High TensorCore idle time percentage of <b>{}%</b> indicates that TensorCores are spending a significant amount of time waiting, not working. Please consider the following optimizations: </p><ul><li><b>Reduce Kernel Launch Overhead:</b> Batch small operations into larger ones to reduce the number of kernel launches.</li><li><b>Minimize Python Overhead:</b> Enclose more operations within compiled graphs or functions to reduce Python interpreter overhead between steps.</li><li><b>Avoid Small CPU Ops:</b> Shift small, frequent operations from the CPU to the device if possible.</li><li><b>Use Asynchronous Operations:</b> Employ asynchronous execution for tasks like checkpointing or metric logging to prevent blocking device execution.</li></ul>",
        one(tpu_percent(data, false)?)
    ))
}

fn sparse_core_bound(data: &dyn ToolData) -> Data<String> {
    Ok(format!(
        "<p>Your program is likely bottlenecked by <b>SparseCore Operations</b> in the TPU: <b>{}% of the total step time </b> is spent on SparseCore. Please consider the following optimizations: </p><ul><li><b>Refine Sparse Data Representation:</b> Ensure your sparse tensors are in the most performant format for your hardware (e.g., CSR/CSC if suitable). Pre-process data to improve memory access patterns on the SparseCore, like sorting indices or grouping related features.</li><li><b>Streamline Embedding Tables:</b> For large embedding tables, consider quantization (reducing precision like int8) or pruning to significantly cut down their memory footprint and processing load on the SparseCore.</li><li><b>Utilize Framework-Specific Sparse APIs:</b> Employ specialized APIs designed for sparse operations on your platform (e.g., tf.tpu.experimental.embedding.TPU Embedding for TensorFlow/TPU). These are highly optimized for direct SparseCore interaction.</li></ul>",
        one(tpu_percent(data, true)?)
    ))
}

fn sparse_core_offload(data: &dyn ToolData) -> Data<String> {
    let (async_done, memory) = (async_done_percent(data)?, peak_memory_percent(data)?);
    Ok(format!(
        "<p>While your program is offloading work to the SparseCore, it isn't being effectively overlapped with TensorCore work due to the compiler predicting the offloaded work taking less time than it actually does. <b>{}%</b> of time is spent on async-done operations with low memory utilization of <b>{}%</b>. This means the TensorCore is waiting on the offloaded SparseCore operation to complete. Consider increasing the values of the following flags to improve how efficiently SparseCore operations are overlapped with TensorCore operations by the compiler:</p><ul><li><b>xla_tpu_sparse_core_all_gather_latency_multiplier</b>: Cost model scaling factor for SparseCore offloaded all-gather. 1.0 (schedule using XLA cost model) / 0.0 (synchronous scheduling) / inf (schedule on data dependencies).</li><li><b>xla_tpu_sparse_core_all_reduce_latency_multiplier</b>: Cost model scaling factor for SparseCore offloaded all-reduce.</li><li><b>xla_tpu_sparse_core_reduce_scatter_latency_multiplier</b>: Cost model scaling factor for SparseCore offloaded reduce-scatter.</li><li><b>Note:</b> Increasing these values will likely increase the memory usage of your program due to holding intermediate results from SparseCore longer before the TensorCore resumes operations on them.</li></ul>",
        one(async_done),
        one(memory)
    ))
}

pub(crate) const RULES: [Rule; 12] = [
    Rule { name: "BarrierCoresRule", meets: |data| event_percent(data, SPECIAL_OP).is_ok_and(|percent| percent >= SPECIAL_OP_PERCENT), generate: barrier_cores },
    Rule { name: "CollectiveBoundRule", meets: |data| collective_percent(data).is_ok_and(|percent| percent >= COLLECTIVE_PERCENT), generate: collective_bound },
    Rule { name: "ComputeBoundRule", meets: |data| matches!((hbm(data), mxu(data)), (Ok(hbm), Ok(mxu)) if mxu > MXU_HIGH && hbm < HBM_LOW), generate: compute_bound },
    Rule { name: "DataShuffleBoundRule", meets: |data| data_shuffle_percent(data).is_ok_and(|percent| percent >= DATA_SHUFFLE_PERCENT), generate: data_shuffle_bound },
    Rule { name: "DataTransferBoundRule", meets: |data| input_bound(data) && percent_of_input(data, true).is_ok_and(|percent| percent >= DATA_TRANSFER_PERCENT), generate: data_transfer_bound },
    Rule { name: "DebugPrintRule", meets: |data| host_event_percents(data, DEBUG_PRINT).is_ok_and(|hosts| hosts.iter().any(|(_, percent)| *percent >= DEBUG_PRINT_PERCENT)), generate: debug_print },
    Rule { name: "HostProcessingBoundRule", meets: |data| input_bound(data) && percent_of_input(data, false).is_ok_and(|percent| percent >= HOST_PROCESSING_PERCENT), generate: host_processing_bound },
    Rule {
        name: "InputBoundRule",
        meets: |data| {
            input_bound(data) && matches!((percent_of_input(data, false), percent_of_input(data, true)), (Ok(other), Ok(enqueue)) if other.abs() < ZERO_EPSILON && enqueue.abs() < ZERO_EPSILON)
        },
        generate: input_bound_text,
    },
    Rule { name: "MemoryBoundRule", meets: |data| matches!((hbm(data), mxu(data)), (Ok(hbm), Ok(mxu)) if hbm > HBM_HIGH && mxu < MXU_LOW), generate: memory_bound },
    Rule { name: "TensorCoreIdleBoundRule", meets: |data| latency_bound(data) && tpu_percent(data, false).is_ok_and(|percent| percent > TENSOR_CORE_IDLE_PERCENT), generate: tensor_core_idle_bound },
    Rule { name: "SparseCoreBoundRule", meets: |data| latency_bound(data) && tpu_percent(data, true).is_ok_and(|percent| percent > SPARSE_CORE_PERCENT), generate: sparse_core_bound },
    Rule {
        name: "SparseCoreOffloadRule",
        meets: |data| matches!((async_done_percent(data), peak_memory_percent(data)), (Ok(async_done), Ok(memory)) if async_done > ASYNC_DONE_PERCENT && memory < MEMORY_HIGH),
        generate: sparse_core_offload,
    },
];

pub fn run(data: &dyn ToolData, registered: &[&str]) -> Data<Vec<Suggestion>> {
    let mut suggestions = Vec::new();
    for rule in RULES.iter().filter(|rule| registered.contains(&rule.name)) {
        if (rule.meets)(data) {
            suggestions.push(Suggestion { rule: rule.name, text: (rule.generate)(data)? });
        }
    }
    Ok(suggestions)
}

pub fn render(suggestions: &[Suggestion]) -> String {
    let mut out = String::from("{\"suggestions\":[");
    for (index, suggestion) in suggestions.iter().enumerate() {
        out.push_str(if index > 0 { ",{\"ruleName\":" } else { "{\"ruleName\":" });
        json_string(&mut out, suggestion.rule);
        out.push_str(",\"suggestionText\":");
        json_string(&mut out, &suggestion.text);
        out.push('}');
    }
    out.push_str("]}");
    out
}

struct Session {
    fractions: Option<Results>,
}

impl ToolData for Session {
    fn event_fractions(&self, event: &str) -> Data<Fractions> {
        let fractions = self.fractions.as_ref().ok_or_else(|| "Can't parse an xspace of the session as binary proto".to_string())?;
        fractions.get(event).cloned().ok_or_else(|| format!("Event not found: {event}"))
    }
}

fn session_fractions(paths: &[PathBuf]) -> Option<Results> {
    let mut fractions = Results::new();
    for path in paths {
        let map = crate::read_file(path).ok()?;
        let mut planes = crate::parse_checked(&map, false).ok()??;
        crate::finish(&mut planes, &map, true);
        let hostname = crate::xplane::fields(&map).find_map(|(tag, field)| match (tag, field) {
            (4, crate::xplane::Field::Bytes(_, name)) => Some(String::from_utf8_lossy(name).into_owned()),
            _ => None,
        });
        let hostname = hostname.unwrap_or_else(|| crate::host_name(path));
        for event in ANALYZED_EVENTS {
            accumulate(analyze(&planes, &map, &hostname, event), fractions.entry(event.to_string()).or_default());
        }
        crate::release((map, planes));
    }
    Some(fractions)
}

pub fn json(paths: &[PathBuf]) -> Option<String> {
    run(&Session { fractions: session_fractions(paths) }, &THIRD_PARTY_RULES).ok().map(|suggestions| render(&suggestions))
}

#[cfg(test)]
#[path = "../tests/inline/tools/smart_suggestion.rs"]
mod tests;
