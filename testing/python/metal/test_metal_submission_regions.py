"""Ordered device regions share a handoff; host effects stay outside."""

import pytest

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang import tvm
from tilelang.metal.transform import MarkHostMetalContext
from tvm import tirx


def _call(name, stack):
    return tirx.Evaluate(tirx.Call("int32", "tirx.tvm_call_packed_lowered", [tirx.StringImm(name), stack, 0, 1]))


def _marked(make_body, *, symbols=("first", "second")):
    stack = tirx.Var("stack", "handle")
    enabled = tirx.Var("enabled", "int32")
    body = tirx.AttrStmt(0, "compute_scope", "test", make_body(stack, enabled))
    body = tirx.AttrStmt(0, "preserved_outer_annotation", 17, body)
    function = tirx.PrimFunc([stack, enabled], body).with_attr("global_symbol", "main")
    module = tvm.IRModule({"main": function}).with_attr("tl.device_kernel_symbols", list(symbols))
    return MarkHostMetalContext()(module)["main"].body


def _contexts(body):
    result = []
    tirx.stmt_functor.post_order_visit(
        body, lambda node: result.append(node) if isinstance(node, tirx.AttrStmt) and node.attr_key == "metal_context" else None
    )
    return result


def test_device_region_preserves_loops_branches_and_attributes():
    def body(stack, enabled):
        index = tirx.Var("index", "int32")
        calls = tirx.SeqStmt([_call("first", stack), tirx.IfThenElse(enabled > 0, _call("second", stack), None)])
        return tirx.AttrStmt(0, "preserved_inner_annotation", 19, tirx.For(index, 0, 3, tirx.ForKind.SERIAL, calls))

    marked = _marked(body)
    assert marked.attr_key == "preserved_outer_annotation"
    contexts = _contexts(marked)
    assert len(contexts) == 1
    assert marked.body.attr_key == "compute_scope"
    assert contexts[0].body.attr_key == "preserved_inner_annotation"
    assert isinstance(contexts[0].body.body, tirx.For)
    assert isinstance(contexts[0].body.body.body, tirx.SeqStmt)


@pytest.mark.parametrize("effect", ["callback", "return", "external"])
def test_host_effects_are_not_moved_into_device_region(effect):
    def body(stack, enabled):
        if effect == "callback":
            middle = _call("host_callback", stack)
        elif effect == "return":
            middle = tirx.IfThenElse(enabled > 0, tirx.Evaluate(tirx.Call("int32", "tirx.ret", [0])), None)
        else:
            middle = tirx.Evaluate(tirx.call_extern("int32", "host_external", enabled))
        return tirx.SeqStmt([_call("first", stack), middle, _call("second", stack)])

    contexts = _contexts(_marked(body))
    assert len(contexts) == 2
    assert [context.body.value.args[0].value for context in contexts] == ["first", "second"]


def test_no_device_symbols_means_no_host_callback_is_reclassified():
    assert not _contexts(_marked(lambda stack, _: _call("first", stack), symbols=()))


def test_existing_submission_context_is_not_nested():
    marked = _marked(lambda stack, _: tirx.AttrStmt(0, "metal_context", "", _call("first", stack)))
    assert len(_contexts(marked)) == 1


@pytest.mark.parametrize("inside_compute", [False, True])
def test_deep_host_annotations_do_not_recurse_through_python(inside_compute):
    import sys

    stack = tirx.Var("stack", "handle")
    # The callback makes the compute region unsafe to batch, so the traversal
    # must reach individual launches even beneath all retained annotations.
    body = tirx.SeqStmt([_call("first", stack), _call("host_callback", stack), _call("second", stack)])
    if not inside_compute:
        body = tirx.AttrStmt(0, "compute_scope", "deep", body)
    depth = 1024
    for index in range(depth):
        body = tirx.AttrStmt(0, "retained_annotation", index, tirx.SeqStmt([_call("host_callback", stack), body]))
    if inside_compute:
        body = tirx.AttrStmt(0, "compute_scope", "deep", body)
    function = tirx.PrimFunc([stack], body)
    module = tvm.IRModule({"main": function}).with_attr("tl.device_kernel_symbols", ["first", "second"])
    recursion_limit = sys.getrecursionlimit()
    marked = MarkHostMetalContext()(module)["main"].body
    assert sys.getrecursionlimit() == recursion_limit
    contexts = _contexts(marked)
    assert [context.body.value.args[0].value for context in contexts] == ["first", "second"]
    if inside_compute:
        marked = marked.body
    for index in reversed(range(depth)):
        assert marked.attr_key == "retained_annotation"
        assert int(marked.value) == index
        assert isinstance(marked.body, tirx.SeqStmt)
        marked = marked.body.seq[1]


def test_compute_scope_does_not_leak_to_following_host_calls():
    stack = tirx.Var("stack", "handle")
    inner = tirx.AttrStmt(0, "compute_scope", "inner", _call("second", stack))
    outer = tirx.AttrStmt(
        0, "compute_scope", "outer", tirx.SeqStmt([_call("first", stack), _call("host_callback", stack), inner, _call("first", stack)])
    )
    outside = _call("first", stack)
    body = tirx.SeqStmt([outer, outside])
    module = tvm.IRModule({"main": tirx.PrimFunc([stack], body)}).with_attr("tl.device_kernel_symbols", ["first", "second"])
    marked = MarkHostMetalContext()(module)["main"].body
    assert len(_contexts(marked)) == 3
    assert marked.seq[1].same_as(outside)


def conditional_program():
    @T.prim_func
    def main(
        X: T.Tensor((256,), "float32"),
        M: T.Tensor((256,), "float32"),
        Y: T.Tensor((256,), "float32"),
        Z: T.Tensor((256,), "float32"),
        enabled: T.int32,
    ):
        with T.Kernel(4, threads=64) as block:
            i = block * 64 + T.get_thread_binding(0)
            M[i] = X[i] + 1
        if enabled != 0:
            with T.Kernel(4, threads=64) as block:
                i = block * 64 + T.get_thread_binding(0)
                Y[i] = M[i] * 3
        with T.Kernel(4, threads=64) as block:
            i = block * 64 + T.get_thread_binding(0)
            Z[i] = T.if_then_else(enabled != 0, Y[i], M[i]) * 2

    return main


def test_lowered_composed_host_has_one_scoped_handoff():
    target = tvm.target.Target("metal", tvm.target.Target("c"))
    with target, tvm.transform.PassContext():
        artifact = tilelang.lower(
            conditional_program(), target=target, target_host="c", enable_host_codegen=True, enable_device_compile=False
        )
    source = artifact.host_mod.inspect_source()
    assert source.count("dispatch_sync(") == 1
    assert source.count('GetGlobal("metal.BeginProgram")') == 1
    assert ".cast<tvm::ffi::Module>()" in source
    assert "[&]() noexcept -> int" in source
    assert "std::rethrow_exception" in source
    assert "TVMFFIErrorMoveFromRaised" in source


@tilelang.testing.requires_metal
def test_dynamic_submission_and_partial_failure_preserve_order():
    import torch
    from tilelang.backend import kernel_capture

    kernel = tilelang.compile(conditional_program(), target="metal", execution_backend="tvm_ffi", out_idx=[])
    source = torch.arange(256, dtype=torch.float32, device="mps")
    middle, branch, output = (torch.full_like(source, -7) for _ in range(3))
    bound = kernel.bind({0: source, 1: middle, 2: branch, 3: output}, (4,))
    active = kernel_capture(tvm.target.Target("metal"), max_kernels=1)
    if active is None:
        pytest.skip("device does not expose native timestamp counters")
    torch.mps.synchronize()
    try:
        active.start()
        with pytest.raises(Exception, match="Kernel timestamp capacity exhausted"):
            bound(1)
        torch.mps.synchronize()
        torch.testing.assert_close(middle.cpu(), torch.arange(256) + 1, check_dtype=False)
        # The failed second launch must not reach the third launch.
        assert torch.equal(branch.cpu(), torch.full((256,), -7.0))
        assert torch.equal(output.cpu(), torch.full((256,), -7.0))
        with pytest.raises(Exception, match="partial timestamps are not valid"):
            active.finish()
    finally:
        active.close()
        torch.mps.synchronize()
    for enabled, count, factor in ((0, 2, 2), (1, 3, 6)):
        active = kernel_capture(tvm.target.Target("metal"), max_kernels=3)
        try:
            active.start()
            bound(enabled)
            torch.mps.synchronize()
            assert len(active.finish()) == count
            torch.testing.assert_close(output.cpu(), (torch.arange(256) + 1) * factor, check_dtype=False)
        finally:
            active.close()


@tilelang.testing.requires_metal
def test_composed_pass_finishes_before_next_host_operation():
    import torch

    kernel = tilelang.compile(conditional_program(), target="metal", execution_backend="tvm_ffi", out_idx=[])
    source = torch.arange(256, dtype=torch.float32, device="mps")
    middle, branch, output = (torch.empty_like(source) for _ in range(3))
    bound = kernel.bind({0: source, 1: middle, 2: branch, 3: output}, (4,))
    for enabled, factor in ((1, 6), (0, 2), (1, 6)):
        bound(enabled)
        # This copy begins another encoder on Torch's command buffer. The
        # program must already have closed its own pass without committing it.
        copied = output.clone()
        torch.mps.synchronize()
        torch.testing.assert_close(copied.cpu(), (torch.arange(256) + 1) * factor,
                                   check_dtype=False)


def test_host_callback_splits_contiguous_launch_groups():
    def body(stack, _):
        return tirx.SeqStmt([_call("first", stack), _call("second", stack),
                             _call("host_callback", stack),
                             _call("first", stack), _call("second", stack)])

    marked = _marked(body)
    contexts = _contexts(marked)
    assert len(contexts) == 2
    assert all(isinstance(context.body, tirx.SeqStmt) and len(context.body.seq) == 2
               for context in contexts)
    sequence = marked.body.body
    assert sequence.seq[1].value.args[0].value == "host_callback"
