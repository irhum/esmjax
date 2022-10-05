import dataclasses

from typing import List, Mapping, Optional, Tuple

from flax import linen as nn, traverse_util
from flax.core import frozen_dict
from flax.linen import partitioning as nn_partitioning

# Default sharding rules for the ESM-2 model on TPUs
DEFAULT_TPU_RULES = [
    ("batch", "X"),
    ("hidden", "Y"),
    ("heads", "Y"),
    ("embed_kernel", "X"),
    ("embed", "Y"),
]


def get_params_axes(
    esm_params: frozen_dict.FrozenDict,
    esm_axes: frozen_dict.FrozenDict,
    rules: List[Tuple[str, str]] = DEFAULT_TPU_RULES,
):
    """Converts `esm_axes` to a PyTree with the same structure as `esm_params`
    and its named axes to the logical axes of the mesh.

    Args:
        esm_params (frozen_dict.FrozenDict): Params dict for ESM2. Could be
            the output of `jax.eval_shape`, since only the *structure* of the
            PyTree is used.
        esm_axes (frozen_dict.FrozenDict): The `params_axes` dict obtained from `.init`
        rules (List[Tuple[str, str]], optional): Rules to convert named axes to logical mesh axes.
            Defaults to DEFAULT_TPU_RULES.

    Returns:
        axes_dict: A PyTree with same structure as `esm_params`, with sharding pattern
            for *all* params. (All with no sharding originally now have None, that is
            full replication on all devices in mesh.)
    """
    axes_modifier_dict = nn_partitioning.get_axis_names(esm_axes)
    axes_modifier_dict = traverse_util.flatten_dict(axes_modifier_dict, sep="/")

    axes_dict = traverse_util.flatten_dict(esm_params["params"], sep="/")
    axes_dict = {
        k: nn_partitioning.logical_to_mesh_axes(axes_modifier_dict[k], rules=rules)
        if k in axes_modifier_dict.keys()
        else None
        for k in axes_dict.keys()
    }
    axes_dict = traverse_util.unflatten_dict(axes_dict, sep="/")
    axes_dict = frozen_dict.freeze({"params": axes_dict})
    return axes_dict


@dataclasses.dataclass
class ShardMixIn:
    """Adds parameter sharding constraints for any flax.linen Module.

    This is a mix-in class that overrides the `param` method of the
    original Module, to selectively add sharding constraints as specified
    in `shard_axes`"""

    shard_axes: Optional[Mapping[str, Tuple[str, ...]]] = None

    # Modifies off https://github.com/google/flax/blob/main/flax/linen/partitioning.py#L304
    def param(self, name: str, *init_args):
        # Initialize using the original Module's `param` method
        param = super().param(name, *init_args)

        # If `shard_axes` specified and param name in the dict, apply constraint
        if self.shard_axes and (name in self.shard_axes.keys()):
            axes = self.shard_axes[name]

            # Apply the sharding constraint (e.g. axes=('embedding', 'hidden'))
            param = nn_partitioning.with_sharding_constraint(param, axes)

            # Sow this, to have the AxisMetadata available at initialization.
            self.sow(
                "params_axes",
                f"{name}_axes",
                nn_partitioning.AxisMetadata(axes),
                reduce_fn=nn_partitioning._param_with_axes_sow_reduce_fn,
            )

        return param


# Just the original Flax layers, with the mix-in inherited.
# No need to write my own layer definitions for these!
class DenseGeneral(ShardMixIn, nn.DenseGeneral):
    pass


class Dense(ShardMixIn, nn.Dense):
    pass
