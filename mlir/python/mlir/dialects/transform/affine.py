#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from .._affine_transform_ops_gen import *
from .._affine_transform_ops_gen import _Dialect

try:
    from ...ir import *
    from ...dialects import transform
    from .._ods_common import _cext as _ods_cext
except ImportError as e:
    raise RuntimeError("Error loading imports from extension module") from e

from typing import Optional, Union, overload


@_ods_cext.register_operation(_Dialect, replace=True)
class ParallelizeOp(ParallelizeOp):
    """Specialization for ParallelizeOp."""

    @overload
    def __init__(
        self,
        parallel_type: Type,
        target: Union[Operation, OpView, Value],
        *,
        loc=None,
        ip=None,
    ): ...

    @overload
    def __init__(
        self,
        target: Union[Operation, OpView, Value],
        *,
        loc=None,
        ip=None,
    ): ...

    def __init__(
        self,
        parallel_type_or_target: Union[Operation, OpView, Type, Value],
        target_or_none: Optional[Union[Operation, OpView, Value]] = None,
        *,
        loc=None,
        ip=None,
    ):
        if isinstance(parallel_type_or_target, Type):
            parallel_type = parallel_type_or_target
            target = target_or_none
        else:
            parallel_type = transform.OperationType.get("affine.parallel")
            target = parallel_type_or_target

        super().__init__(parallel_type, target, loc=loc, ip=ip)


@_ods_cext.register_operation(_Dialect, replace=True)
class TileOp(TileOp):
    """Specialization for TileOp."""

    def __init__(
        self,
        target: Union[Operation, OpView, Value],
        dimensions,
        tile_sizes,
        *,
        point_dimensions=None,
        loc=None,
        ip=None,
    ):
        loop_type = transform.OperationType.get("affine.for")
        super().__init__(
            loop_type,
            loop_type,
            target,
            dimensions,
            tile_sizes,
            point_dimensions=point_dimensions,
            loc=loc,
            ip=ip,
        )
