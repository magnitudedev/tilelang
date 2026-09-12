/* Native partial application for packed runtime functions. */
#include "support/check.h"

#include <tvm/ffi/container/array.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>

#include <cstdint>
#include <vector>

namespace tvm {
namespace tl {

using namespace ffi;

TVM_FFI_STATIC_INIT_BLOCK() {
  reflection::GlobalDef().def_packed(
      "tilelang.runtime.bind_packed_function", [](PackedArgs args, Any *ret) {
        ICHECK(args.size() == 5)
            << "bind_packed_function expects function, parameter count, static "
               "indices, static values, and dynamic indices";
        Function callee = args[0].cast<Function>();
        int64_t parameter_count = args[1].cast<int64_t>();
        Array<int64_t> static_indices = args[2].cast<Array<int64_t>>();
        Array<Any> static_values = args[3].cast<Array<Any>>();
        Array<int64_t> dynamic_indices = args[4].cast<Array<int64_t>>();

        ICHECK(parameter_count >= 0);
        ICHECK(static_indices.size() == static_values.size());
        std::vector<bool> occupied(parameter_count, false);
        for (int64_t index : static_indices) {
          ICHECK(index >= 0 && index < parameter_count);
          ICHECK(!occupied[index]);
          occupied[index] = true;
        }
        for (int64_t index : dynamic_indices) {
          ICHECK(index >= 0 && index < parameter_count);
          ICHECK(!occupied[index]);
          occupied[index] = true;
        }
        for (bool slot : occupied) ICHECK(slot);

        *ret = Function::FromPacked(
            [callee, parameter_count, static_indices, static_values,
             dynamic_indices](PackedArgs dynamic_args, Any *result) {
              ICHECK(dynamic_args.size() == dynamic_indices.size())
                  << "bound function received the wrong number of dynamic arguments";
              std::vector<AnyView> frame(parameter_count);
              for (int64_t i = 0; i < static_indices.size(); ++i) {
                frame[static_indices[i]] = static_values[i];
              }
              for (int64_t i = 0; i < dynamic_indices.size(); ++i) {
                frame[dynamic_indices[i]] = dynamic_args[i];
              }
              callee.CallPacked(frame.data(), static_cast<int32_t>(frame.size()),
                                result);
            });
      });
}

}  // namespace tl
}  // namespace tvm
