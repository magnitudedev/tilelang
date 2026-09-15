"""Submit maximal device-only host regions on the MPS queue."""

from tvm import tirx as tir
from tvm.ir import Op
from tvm.tirx import AttrStmt
from tvm.tirx.transform import prim_func_pass


_packed = Op.get("tirx.tvm_call_packed_lowered")
# These operations only prepare the generated host launcher's argument frame.
# Arbitrary calls, including host callbacks and returns, remain boundaries.
_frame_ops = frozenset(("tirx.tvm_struct_get", "tirx.tvm_struct_set",
                        "tirx.tvm_stack_alloca", "tirx.tvm_stack_make_array",
                        "tirx.tvm_stack_make_shape"))


def _device_launch(call, symbols):
    return (isinstance(call, tir.Call) and call.op.same_as(_packed)
            and isinstance(call.args[0], tir.StringImm)
            and call.args[0].value in symbols)


def _mark_contexts(body, symbols):
    # Both traversals run natively: large host programs can contain thousands of
    # nested scopes, so Python recursion must not follow the IR depth. Monotone
    # counters classify entire subtrees without rewalking their descendants.
    starts = {}
    safe = set()
    pure = set()
    effects = launches = compute_depth = 0

    def inspect_enter(node):
        nonlocal effects, launches, compute_depth
        starts[node] = effects, launches, compute_depth
        if isinstance(node, AttrStmt):
            if node.attr_key == "metal_context":
                effects += 1
                starts.pop(node)
                return node
            if node.attr_key == "compute_scope":
                compute_depth += 1
        elif isinstance(node, tir.Call):
            if _device_launch(node, symbols):
                launches += 1
            elif isinstance(node.op, Op):
                effect = node.op.get_attr("TCallEffectKind")
                if node.op.name not in _frame_ops and (effect is None or int(effect) not in (0, 1)):
                    effects += 1
            else:
                effects += 1
        elif isinstance(node, tir.BufferStore):
            effects += 1
        return None

    def inspect_leave(node):
        nonlocal compute_depth
        before_effects, before_launches, depth = starts.pop(node)
        if (isinstance(node, tir.Stmt) and depth and effects == before_effects
                and not (isinstance(node, AttrStmt) and node.attr_key == "compute_scope")):
            pure.add(node)
            if launches > before_launches:
                safe.add(node)
        if isinstance(node, AttrStmt) and node.attr_key == "compute_scope":
            compute_depth -= 1
        return None

    tir.stmt_functor.ir_transform(body, inspect_enter, inspect_leave)

    created = set()

    def mark(node):
        if isinstance(node, AttrStmt) and node.attr_key == "metal_context":
            return node
        if isinstance(node, tir.Stmt) and node in safe:
            marked = AttrStmt(0, "metal_context", "", node)
            created.add(marked)
            return marked
        return None

    def join_regions(node):
        if not isinstance(node, tir.SeqStmt):
            return None
        result, pending = [], []
        launches_pending = False

        def flush():
            nonlocal launches_pending
            if launches_pending:
                region = pending[0] if len(pending) == 1 else tir.SeqStmt(pending)
                marked = AttrStmt(0, "metal_context", "", region)
                created.add(marked)
                result.append(marked)
            else:
                result.extend(pending)
            pending.clear()
            launches_pending = False

        for statement in node.seq:
            if statement in created:
                pending.append(statement.body)
                launches_pending = True
            elif statement in pure:
                pending.append(statement)
            else:
                flush()
                result.append(statement)
        flush()
        return result[0] if len(result) == 1 else tir.SeqStmt(result)

    return tir.stmt_functor.ir_transform(body, mark, join_regions)


def MarkHostMetalContext():
    def pass_fn(func, mod, ctx):
        symbols = frozenset(str(symbol) for symbol in (mod.attrs.get("tl.device_kernel_symbols", ()) if mod.attrs else ()))
        if not symbols:
            return func
        return func.with_body(_mark_contexts(func.body, symbols), span=func.span)

    return prim_func_pass(pass_fn, opt_level=0)


__all__ = ["MarkHostMetalContext"]
