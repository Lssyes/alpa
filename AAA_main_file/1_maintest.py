from typing import Any
import alpa
from alpa.testing import assert_allclose
import copy
from flax import linen as nn
from flax.training.train_state import TrainState
import jax
import jax.numpy as jnp
from jax import random
import optax
import ray
from alpa.util import benchmark_func, trace_jaxpr_with_micro_batch, print_jaxpr_computation_graph

alpa.util.disable_tqdm_globally()
a = alpa.init(cluster="ray")




class MaunulModelPipeline(nn.Module):
    hidden_dim: int
    
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=self.hidden_dim)(x)
        # x = nn.Dense(features=self.hidden_dim)(x)
        # x = nn.Dense(features=self.hidden_dim)(x)
        # x = nn.relu(x)
        # x = nn.Dense(features=self.hidden_dim)(x)
        # x = nn.relu(x)
        # x = nn.Dense(features=self.hidden_dim)(x)
        # x = nn.Dense(features=self.hidden_dim)(x)
        # x = nn.Dense(features=self.hidden_dim)(x)
        # x = nn.relu(x)
        # x = nn.Dense(features=self.hidden_dim)(x)
        # x = nn.relu(x)
        return x



dim = 8
batch_size = 512

# Generate ground truth W and b
rngkey = jax.random.PRNGKey(0)
k1, k2 = random.split(rngkey)
W = random.normal(k1, (dim, dim))
b = random.normal(k2, (dim,))

# Generate the training data
ksample, knoise = random.split(k1)
x = random.normal(ksample, (batch_size, dim))
y = (x @ W + b) + 0.1 * random.normal(knoise, (batch_size, dim))

# Initialize a train state, which includes the model paramter and optimizer
# state.
model = MaunulModelPipeline(hidden_dim=dim)
params = model.init(rngkey, x)
tx = optax.adam(learning_rate=1e-3)
tx = optax.sgd(learning_rate=1e-3)
state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)


print(params)
print(jax.tree_util.tree_map(lambda x: x.shape, params))




# def jax_trainstep(state, batch):
#     def loss(params):
#         out = state.apply_fn(params, batch["x"])
#         loss = jnp.mean((batch["y"] - out)**2)
#         return loss
#     grads = jax.grad(loss)(state.params)
#     new_state = state.apply_gradients(grads=grads)
#     return new_state







# @alpa.parallelize(method=alpa.PipeshardParallel(num_micro_batches=16,
#                                                 layer_option=alpa.AutoLayerOption(layer_num=1),
#                                                 stage_option="auto"))
# def MaunulModelPipeline_TrainStep(state, batch):
#     def loss(params):
#         out = state.apply_fn(params, batch["x"])
#         loss = jnp.mean((batch["y"] - out)**2)
#         return loss
#     grads = alpa.grad(loss)(state.params) #####  这个地方切换 jax.grad
#     new_state = state.apply_gradients(grads=grads)
#     return new_state

fake_params = [jnp.ones((dim, dim)) for _ in range(64)]

@alpa.parallelize(method=alpa.PipeshardParallel(num_micro_batches=16,
                                                layer_option=alpa.AutoLayerOption(layer_num=6),
                                                stage_option="auto"))
def fake_Trainstep(fake_params, batch):
    def loss(fake_params):
        # out = state.apply_fn(params, batch["x"])
        out = jax.lax.dot(batch["x"], fake_params[0])
        for i, param in enumerate(fake_params):
            if(i==0):
                continue
            out = jax.lax.dot(out, param)
        loss = jnp.mean((batch["y"] - out)**2)
        return loss
    alpa.grad(loss)(fake_params) #####  这个地方切换 jax.grad
    # new_state = state.apply_gradients(grads=grads)
    return 0


batch = {"x": x, "y": y}


fake_Trainstep(fake_params, batch)


print()

import time
import numpy as np
def alpa_execution():
    global state
    # state = MaunulModelPipeline_TrainStep(state, batch)
def sync_func():
    # jax.local_devices()[0].synchronize_all_activity()
    jax.device_put(0.).block_until_ready()

alpa_costs = benchmark_func(alpa_execution,  warmup=5, number=10, repeat=5) * 1e3
print(f"Alpa execution time.   Mean: {np.mean(alpa_costs):.2f} ms, Std: {np.std(alpa_costs):.2f} ms")

alpa.shutdown()
