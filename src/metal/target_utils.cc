/*!
 * \file tl/metal/target_utils.cc
 * \brief Metal target attribute helpers.
 */

#include "metal/target_utils.h"

#include <tvm/ffi/reflection/registry.h>

#include "dlpack/dlpack.h"

namespace tvm {
namespace tl {

bool TargetIsMetal(Target target) {
  return target->GetTargetDeviceType() == kDLMetal;
}

int TargetMetalGetWarpSize(Target target) {
  (void)target;
  return 32;
}

bool TargetMetalSupportsMetal4(Target target) {
  return target->GetAttr<Bool>("supports_metal4").value_or(Bool(false));
}

bool TargetMetalSupportsSIMDGroupMatrix(Target target) {
  return target->GetAttr<Bool>("supports_simdgroup_matrix")
      .value_or(Bool(false));
}

bool TargetMetalSupportsBFloat16(Target target) {
  return target->GetAttr<Bool>("supports_bfloat16").value_or(Bool(false));
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("tl.TargetIsMetal",
           [](Target target) { return TargetIsMetal(target); })
      .def("tl.TargetMetalGetWarpSize",
           [](Target target) { return TargetMetalGetWarpSize(target); })
      .def("tl.TargetMetalSupportsMetal4",
           [](Target target) { return TargetMetalSupportsMetal4(target); })
      .def("tl.TargetMetalSupportsSIMDGroupMatrix",
           [](Target target) { return TargetMetalSupportsSIMDGroupMatrix(target); })
      .def("tl.TargetMetalSupportsBFloat16",
           [](Target target) { return TargetMetalSupportsBFloat16(target); });
}

} // namespace tl
} // namespace tvm
