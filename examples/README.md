# Examples

## Train a CLIP model with overlapped communication

`clip_overlap.py` trains a CLIP model from [jimm](https://github.com/pythoncrazy/jimm) on a TPU slice. The model uses fully sharded data parallelism (FSDP). Each step does these transfers:

- **Device to device:** an all-gather of the weights for each layer and a gradient reduction. Also an all-gather of the image and text features for the contrastive loss.
- **Host to device:** a copy of the image batch (`uint8`) and the tokens for each step.

The script then writes a profile. `overlap_report.py` reads that profile with the `xprof-rs` command line and shows how much communication the compute hides.

```sh
python examples/clip_overlap.py --logdir /tmp/clip_overlap
python examples/overlap_report.py /tmp/clip_overlap/plugins/profile/<run> --xprof-rs target/release/xprof-rs
xprof-rs --logdir /tmp/clip_overlap
```

Use `--model openai/clip-vit-large-patch14 --per-device-batch 8` for more communication. Use `--no-overlap --prefetch 0` for the slow baseline.

### What the profile showed

I made these changes one at a time. After each change, I read the profile with `xprof-rs`. The numbers come from a TPU v4-8 (4 devices, CLIP B/16, 16 images for each device).

| Step | Change | Step time | Device busy | Communication hidden |
| --- | --- | --- | --- | --- |
| 0 | Baseline: jimm with an automatic mesh, input made in the loop | 140 ms | 23% | 0% |
| 1 | Explicit mesh axis `fsdp` and a loss that gathers the features. XLA does not do tensor-parallel all-reduces on activations after this change. The batches are random. A thread copies them (`--prefetch`). | 43 ms | 77% | 0% |
| 2 | libtpu flags (`--overlap`) start the all-gathers early | 39 ms | 71% | 53% |
| 3 | `jax.jit` on the split state. `nnx.jit` used 8 ms of host time for each step. | 28 ms | 99.5% | 54% |
| 4 | Move the state to its final sharding before the first step. The step compiled two times before. | 28 ms | 99.5% | 54% |

How to find each problem with `xprof-rs`:

- `check_host_boundness` and `get_kpi_metrics` show the duty cycle (the part of the time the device works). A low value means that the host is too slow.
- `list_xplane_events` shows a gap before the first operation of each step. The gap was the host time of `nnx.jit`.
- `list_xplane_events --event_regex=PjitFunction` and `JAX_LOG_COMPILES=1` show a second compile of `jit(step_fn)`. The input state had other shardings than the output state of the step.
- `get_top_hlo_ops` and `overlap_report.py` show which collectives stay in the `XLA Ops` line (exposed) and which move to `Async XLA Ops` (hidden).

`--cache-dir` stores the compiled programs. A second run starts in about 40 seconds and does not compile again.

### Limits

- On TPU v4, libtpu supports asynchronous collective fusion for all-gathers only. libtpu rejects the flags for all-reduce and reduce-scatter on this chip ("not supported on platforms other than Viperlite"). The gradient reductions stay exposed. This is why the hidden part stops near 50%.
- The compute hides the all-gather of the weights best when each layer has enough compute. The larger CLIP L/14 model hides 47% of 24 ms of all-gather per step. B/16 hides 54% of 11 ms.
- The images are random. The loader copies four prepared batches again and again. A real loader must decode images on other threads. Keep the copy in a thread, as `Batches` does.
