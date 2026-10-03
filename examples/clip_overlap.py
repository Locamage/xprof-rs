import argparse
import os
import queue
import threading
import time

parser = argparse.ArgumentParser(description="Train CLIP from jimm with FSDP on all local devices and write an XProf trace.")
parser.add_argument("--model", default="openai/clip-vit-base-patch16")
parser.add_argument("--per-device-batch", type=int, default=16)
parser.add_argument("--steps", type=int, default=8)
parser.add_argument("--warmup", type=int, default=3)
parser.add_argument("--prefetch", type=int, default=2, help="Batches that are copied to the devices before they are needed. 0 copies each batch in the training loop.")
parser.add_argument("--overlap", action=argparse.BooleanOptionalAction, default=True, help="Use the libtpu flags that overlap collectives with compute.")
parser.add_argument("--remat", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--logdir", default="/tmp/clip_overlap")
parser.add_argument("--cache-dir", default="/tmp/clip_overlap_cache", help="Directory for the compiled programs. A second run does not compile again.")
args = parser.parse_args()

OVERLAP_FLAGS = [
    "--xla_tpu_enable_async_collective_fusion=true",
    "--xla_tpu_enable_async_collective_fusion_multiple_steps=true",
    "--xla_tpu_overlap_compute_collective_tc=true",
    "--xla_enable_async_all_gather=true",
    "--xla_tpu_enable_data_parallel_all_reduce_opt=true",
    "--xla_tpu_data_parallel_opt_different_sized_ops=true",
    "--xla_tpu_enable_megacore_fusion=true",
    "--xla_tpu_enable_windowed_einsum_for_all_gather=true",
    "--xla_tpu_enable_windowed_einsum_for_reduce_scatter=true",
]
if args.overlap:
    os.environ["LIBTPU_INIT_ARGS"] = " ".join([os.environ.get("LIBTPU_INIT_ARGS", ""), *OVERLAP_FLAGS]).strip()

import jax

jax.config.update("jax_compilation_cache_dir", args.cache_dir)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx
from jax.sharding import AxisType, reshard, NamedSharding, PartitionSpec as P
from jimm.models import CLIP
from transformers import CLIPTokenizer

IMAGE_SIZE = 224
TEXT_LENGTH = 77
CAPTIONS = ["a photo of a cat", "a photo of a dog", "a picture of a bird", "an image of a car", "a photo of a tree", "a picture of a house", "an image of a person", "a photo of food"]

devices = jax.devices()
batch_size = args.per_device_batch * len(devices)
mesh = jax.make_mesh((len(devices),), ("fsdp",), axis_types=(AxisType.Explicit,))
jax.set_mesh(mesh)
image_sharding = NamedSharding(mesh, P("fsdp", None, None, None))
text_sharding = NamedSharding(mesh, P("fsdp", None))


def clip_loss(image_features, text_features, scale):
    image_features = image_features / jnp.linalg.norm(image_features, axis=-1, keepdims=True)
    text_features = text_features / jnp.linalg.norm(text_features, axis=-1, keepdims=True)
    all_images, all_texts = reshard(image_features, P()), reshard(text_features, P())
    labels = reshard(jnp.arange(image_features.shape[0]), P("fsdp"))
    image_loss = optax.softmax_cross_entropy_with_integer_labels(jnp.exp(scale) * image_features @ all_texts.T, labels).mean()
    text_loss = optax.softmax_cross_entropy_with_integer_labels(jnp.exp(scale) * text_features @ all_images.T, labels).mean()
    return (image_loss + text_loss) / 2


def step_fn(state, images, texts):
    model, optimizer = nnx.merge(graphdef, state)
    images = images.astype(jnp.bfloat16) * (2.0 / 255.0) - 1.0

    def loss_fn(model):
        return clip_loss(model.encode_image(images, do_projection=True), model.encode_text(texts), model.logit_scale[...])

    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grads)
    return nnx.state((model, optimizer)), loss


@nnx.jit
def build():
    model = CLIP.from_pretrained(args.model, use_pytorch=True, use_gradient_checkpointing=args.remat, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16, rngs=nnx.Rngs(0))
    optimizer = nnx.Optimizer(model, optax.adam(1e-4), wrt=nnx.Param)
    return model, optimizer


class Batches:
    def __init__(self, tokens, depth):
        self.tokens = tokens
        random = np.random.default_rng(0)
        self.pool = [random.integers(0, 255, (batch_size, IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8) for _ in range(4)]
        self.count = 0
        self.queue = queue.Queue(maxsize=max(depth, 1))
        self.depth = depth
        if depth:
            threading.Thread(target=self.fill, daemon=True).start()

    def make(self):
        self.count += 1
        batch = jax.device_put((self.pool[self.count % len(self.pool)], self.tokens), (image_sharding, text_sharding))
        return jax.block_until_ready(batch)

    def fill(self):
        while True:
            self.queue.put(self.make())

    def next(self):
        return self.queue.get() if self.depth else self.make()


model, optimizer = build()
model.train()
graphdef, state = nnx.split((model, optimizer))
train_step = jax.jit(step_fn, donate_argnums=0)
tokenizer = CLIPTokenizer.from_pretrained(args.model)
tokens = tokenizer([CAPTIONS[index % len(CAPTIONS)] for index in range(batch_size)], padding="max_length", truncation=True, max_length=TEXT_LENGTH, return_tensors="np")["input_ids"].astype("int32")
batches = Batches(tokens, args.prefetch)
output_state, _ = jax.eval_shape(train_step, state, *batches.make())
state = jax.tree.map(lambda array, output: jax.device_put(array, NamedSharding(mesh, output.sharding.spec)), state, output_state)

previous = None
for _ in range(args.warmup):
    state, loss = train_step(state, *batches.next())
    jax.block_until_ready(loss)

jax.profiler.start_trace(args.logdir)
start = time.time()
for index in range(args.steps):
    with jax.profiler.StepTraceAnnotation("train", step_num=index):
        state, loss = train_step(state, *batches.next())
        if previous is not None:
            jax.block_until_ready(previous)
        previous = loss
jax.block_until_ready(previous)
elapsed = time.time() - start
jax.profiler.stop_trace()
print(f"devices={len(devices)} batch={batch_size} steps={args.steps} step_ms={1000 * elapsed / args.steps:.1f} loss={float(previous):.4f} overlap={args.overlap} prefetch={args.prefetch}")
