"""Put individual device launches on the MPS submission queue."""

from tvm import tirx as tir
from tvm.ir import Op
from tvm.tirx import AttrStmt, Evaluate
from tvm.tirx.transform import prim_func_pass


_packed = Op.get("tirx.tvm_call_packed_lowered")


def _device_launch(call, symbols):
    return (
        isinstance(call, tir.Call)
        and call.op.same_as(_packed)
        and isinstance(call.args[0], tir.StringImm)
        and call.args[0].value in symbols
    )


def _mark_contexts(body, symbols):
    # Host launchers can have thousands of nested allocation/annotation scopes.
    # A PyStmtExprMutator override holds Python frames across each child visit;
    # IRTransform owns that traversal natively and returns from each callback
    # before visiting children. Preserve scopes rather than flattening the IR.
    compute_depth = 0

    def enter(stmt):
        nonlocal compute_depth
        if isinstance(stmt, AttrStmt):
            if stmt.attr_key == "metal_context":
                return stmt
            if stmt.attr_key == "compute_scope":
                compute_depth += 1
        elif isinstance(stmt, Evaluate) and compute_depth and _device_launch(stmt.value, symbols):
            return AttrStmt(0, "metal_context", "", stmt)
        return None

    def leave(stmt):
        nonlocal compute_depth
        if isinstance(stmt, AttrStmt) and stmt.attr_key == "compute_scope":
            compute_depth -= 1
        return None

    return tir.stmt_functor.ir_transform(body, enter, leave, ["tirx.AttrStmt", "tirx.Evaluate"])


def MarkHostMetalContext():
    def pass_fn(func, mod, ctx):
        symbols = frozenset(str(symbol) for symbol in (mod.attrs.get("tl.device_kernel_symbols", ()) if mod.attrs else ()))
        if not symbols:
            return func
        return func.with_body(_mark_contexts(func.body, symbols), span=func.span)

    return prim_func_pass(pass_fn, opt_level=0)


__all__ = ["MarkHostMetalContext"]
