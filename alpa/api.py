"""Top-level user API."""
from typing import Callable, Optional, Sequence, Union
from colorama import Fore, Back, Style

from jax import linear_util as lu
from jax._src import api, traceback_util
from jax._src.util import HashableFunction
from jax.api_util import (argnums_partial, donation_vector,
                          flatten_fun_nokwargs, rebase_donate_argnums)
from jax.core import AbstractValue
from jax.experimental.maps import FrozenDict
from jax.tree_util import tree_flatten, tree_unflatten, PyTreeDef

from alpa.device_mesh import init_global_cluster, shutdown_global_cluster
from alpa.parallel_method import ParallelMethod, ShardParallel
from alpa.pipeline_parallel.primitive_def import mark_gradient
from alpa.util import (auto_donate_argnums, auto_static_argnums, print_jaxpr_computation_graph,
                       abstractify_with_aval, GradFuncTransformContext)
from alpa.version import check_alpa_jaxlib_version
from jax.interpreters import partial_eval as pe
from jax.core import gensym, AbstractValue, ClosedJaxpr
import jax

traceback_util.register_exclusion(__file__)

is_initialized = False


def init(cluster: str = "ray",
         cluster_address: Optional[str] = None,
         num_nodes: Optional[int] = None,
         num_devices_per_node: Optional[int] = None,
         namespace: Optional[str] = "alpa_default_space"):
    """Initialize the global environment.

    `devices_per_node, num_nodes` are used to specify the number of devices.
    If not specified, the number of devices is determined automatically and
    the whole cluster is used.

    For simplicity, the resource specification is only supported for
    ray cluster.

    Args:
      cluster: The distributed cluster.
        Possible choices: {"local", "ray"}.
        "local" means using all local devices on a single node.
        "ray" means using all devices in a ray cluster.
      cluster_address: Address of the distributed cluster.
        If cluster is "ray", this parameter can be used to specify a different
          address that will be used to initialize the ray cluster.
          E.g., "ray://123.45.67.89:10001". If not specified, "auto" will be
          used instead.
        Ignored if cluster is "local".
      num_nodes: The number of nodes.
      num_devices_per_node: The number of devices per node.
    """
    global is_initialized

    if is_initialized:
        print("is_initialized before")
        return
    is_initialized = True

    init_global_cluster(cluster, cluster_address, num_nodes,
                        num_devices_per_node, namespace)


def shutdown():
    """Shutdown the global environment."""
    global is_initialized
    assert is_initialized is True
    is_initialized = False
    shutdown_global_cluster()

# @parallelize 调用 ParallelizedFunc类 的封装
def parallelize(fun: Optional[Callable] = None,
                *,
                static_argnums: Union[Sequence[int], str] = "auto",
                donate_argnums: Union[Sequence[int], str] = "auto",
                batch_argnums: Union[Sequence[int], str] = (1,),
                method: Optional[ParallelMethod] = None):
    """
    Parallelize a jax function.

    Args:
        fun: 要并行化的函数.
        static_argnums: 与 jax.jit 的 static_argnums 参数相同.
          如果是“auto”, alpa 使用启发式规则来推断。
        donate_argnums: 与 jax.jit 的 static_argnums 参数相同.
          如果是“auto”, alpa 使用启发式规则来推断。
        batch_argnums: 作为 data batch 的参数索引.
          该信息用于将原始数据批次拆分为mini-batch, 以执行 梯度累积 或 pipeline并行。
          Alpa 假设张量的第 0 维是批量维度。
        method: 并行化方法
    """
    check_alpa_jaxlib_version()

    def decorate_fun(fun):
        api._check_callable(fun)  # pylint: disable=protected-access
        nonlocal method           # 表示 method 为上一级函数中的局部变量
        method = method or ShardParallel()  # 若 method 为 None 则用 ShardParallel
        return ParallelizedFunc(fun, static_argnums, donate_argnums, batch_argnums, method)

    if fun is None:
        return decorate_fun
    return decorate_fun(fun)  # ParallelizedFunc()(fun)


# 被 parallelize @ 过的func 就会变成这个Class的形状
# Officely, 经过alpa.parallelize转换后的函数。
class ParallelizedFunc:
    """ 经过alpa.parallelize转换后的函数. """
    def __init__(
        self,
        fun: Callable,
        static_argnums: Union[Sequence[int], str],
        donate_argnums: Union[Sequence[int], str],
        batch_argnums: Union[Sequence[int], str],
        method: ParallelMethod,
    ):
        self.fun = fun
        self.static_argnums = static_argnums
        self.donate_argnums = donate_argnums
        self.batch_argnums = batch_argnums
        self.method = method

        self.last_executable = None

    @traceback_util.api_boundary
    def __call__(self, *args):
        """ 在驱动程序上启动计算。
        Launch the computation on the driver."""
        # print("Cccccccccaaaaaaalllllll  me")
        executable, _, out_tree, args_flat = (self._decode_args_and_get_executable(*args))
        out = executable.launch_on_driver(*args_flat)
        # print(out_tree())
        return tree_unflatten(out_tree(), out)

    def get_executable(self, *args):
        """Get the compiled exectuable."""
        executable, _, _, _ = self._decode_args_and_get_executable(*args)
        return executable

    def preshard_dynamic_args(self, *args):
        """Shard the dynamic arguments."""
        executable, in_tree, _, args_flat = (
            self._decode_args_and_get_executable(*args))
        sharded_args = executable.preshard_dynamic_args(*args_flat)
        return tree_unflatten(in_tree, sharded_args)

    def get_last_executable(self):
        """Return the last compiled executable for this function."""
        return self.last_executable

    def _decode_args_and_get_executable(self, *args):
        """Flatten PyTree arguments and get the executable."""
        static_argnums, donate_argnums, batch_argnums = (self.static_argnums,
                                                         self.donate_argnums,
                                                         self.batch_argnums)
        kwargs = {}
        # print(f"static_argnums, donate_argnums, batch_argnums: {static_argnums, donate_argnums, batch_argnums}")
        f = lu.wrap_init(self.fun)

        # Deal with static arguments and extract dynamic arguments
        if static_argnums == "auto":
            static_argnums = auto_static_argnums(args) 

        if static_argnums:
            dyn_argnums = [
                i for i in range(len(args)) if i not in static_argnums
            ]
            # Freeze static dict to make it hashable
            frozen_args = []
            for i, arg in enumerate(args):
                if i in static_argnums and isinstance(arg, dict):
                    frozen_args.append(FrozenDict(arg))
                else:
                    frozen_args.append(arg)
            f, dyn_args = argnums_partial(f, dyn_argnums, frozen_args)
        else:
            dyn_args = args

        # Flatten pytree arguments
        #   1. args_flat 是 TrainState 和 dict{x, y} 的 PytreeFlat 形式
        #      in_treee  是 TrainState 和 dict{x, y} 的 PyTreeDef, 与上面对应
        #   2. f,out_tree是 jax 封装了的 ？？？？？不理解
        #      
        args_flat, in_tree = tree_flatten(dyn_args)    
        f, out_tree = flatten_fun_nokwargs(f, in_tree)
        # pylint: disable=unnecessary-lambda
        out_tree_hashable = HashableFunction(lambda: out_tree(), closure=None)

        # Deal with donate argnums
        if donate_argnums == "auto":
            donate_argnums = auto_donate_argnums(args)

        donate_tuple = rebase_donate_argnums(donate_argnums, static_argnums)
        if donate_tuple:
            donated_invars = donation_vector(donate_tuple, dyn_args, kwargs)
        else:
            donated_invars = (False,) * len(args_flat)

        # Deal with batch argnums
        batch_tuple = rebase_donate_argnums(batch_argnums, static_argnums)
        batch_invars = donation_vector(batch_tuple, dyn_args, kwargs)

        # Compile
        abstract_args = map(abstractify_with_aval, args_flat)
        
        ## for DEBUG, 通过 遍历输入 利用函数 abstractify_with_aval 转换为抽象值，转换成jaxpr
        ## 这个地方的jaxpr是不可能进行layer constuciton 的因为没有 with GradFuncTransformContext(layer_option.transform):
        # avals = []
        # for arg in args_flat:
        #     avals.append(abstractify_with_aval(arg))        
        # with jax.disable_jit():
        #     jaxpr, _, consts = pe.trace_to_jaxpr_final(f, avals)                
        # closed_jaxpr = ClosedJaxpr(jaxpr, consts)

        # print_jaxpr_computation_graph(closed_jaxpr)
        # print()
                
        
        executable = _compile_parallel_executable(f, in_tree, out_tree_hashable,
                                                  static_argnums,
                                                  donated_invars, batch_invars,
                                                  self.method, *abstract_args)

        self.last_executable = executable
        return executable, in_tree, out_tree, args_flat


@lu.cache
def _compile_parallel_executable(
    fun: lu.WrappedFun,
    in_tree: PyTreeDef,
    out_tree_thunk: Callable[[], PyTreeDef],
    static_argnums: Sequence[int],
    donated_invars: Sequence[bool],
    batch_invars: Sequence[bool],
    method: ParallelMethod,
    *avals: Sequence[AbstractValue],
):
    """Cached parallelized callable."""
    # 把“所有”的 invar 参数打印一下
    print(Back.GREEN + "donated\tbatch\tavals")
    for i in range(len(batch_invars)):
        print(Back.GREEN + f"{donated_invars[i]}\t{batch_invars[i]}\t{avals[i]}")
    print(Style.RESET_ALL)
    
    # Clean stores fo the next call
    for store in fun.stores:
        if store:
            store.reset()
    batch_invars = list(batch_invars)
    for idx, aval in enumerate(avals):
        if len(aval.shape) == 0:
            batch_invars[idx] = False
    batch_invars = tuple(batch_invars)

    # Compile a callable
    return method.compile_executable(fun, 
                                     in_tree, 
                                     out_tree_thunk,
                                     static_argnums, 
                                     donated_invars,
                                     batch_invars, 
                                     *avals)


def clear_executable_cache():
    """Clear all cached executables."""
    _compile_parallel_executable.cache_clear()


def grad(*args, **kwargs):
    """This is the same as jax.grad, except that alpa inserts a
    gradient marker after the gradient computation.

    This function annotates all gradient tensors. This information is used to
    perform gradient accumulation transformation.
    此函数对所有梯度张量进行注释。该信息用于执行 gradient accumulation 变换。
    If any auxiliary tensors are returned, they are averaged over mini batches
    in the same way as how the gradients are averaged.
    如果返回任何 辅助tensor, 则以与梯度平均相同的方式在 mini batches 上对其进行平均。
    """

    def ret(*call_args, **call_kwargs):
        # Apply transformations (e.g., layer construction, rematerialization)
        # to the forward func
        arg_list = list(args)
        for transform in GradFuncTransformContext.transforms:
            arg_list[0] = transform(arg_list[0])
        grad_func = api.grad(*arg_list, **kwargs)

        grads = grad_func(*call_args, **call_kwargs)
        return mark_gradient(grads)

    return ret


def value_and_grad(*args, **kwargs):
    """This is the same as jax.value_and_grad, except that alpa inserts a
    gradient marker after the gradient computation.


    This function annotates all gradient tensors. This information is used to
    perform gradient accumulation transformation.
    If any auxiliary tensors are returned, they are averaged over mini batches
    in the same way as how the gradients are averaged.
    """

    def ret(*call_args, **call_kwargs):
        # Apply transformations (e.g., layer construction, rematerialization)
        # to the forward func
        arg_list = list(args)
        for transform in GradFuncTransformContext.transforms:
            arg_list[0] = transform(arg_list[0])
        grad_func = api.value_and_grad(*arg_list, **kwargs)

        val, grads = grad_func(*call_args, **call_kwargs)
        return mark_gradient((val, grads))

    return ret
