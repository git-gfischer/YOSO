if(NOT DEFINED TRT_ROOT)
  set(TRT_ROOT "/usr/local/tensorrt" CACHE PATH "TensorRT root directory")
endif()

find_package(CUDAToolkit REQUIRED)

find_path(TRT_INCLUDE_DIR NvInfer.h
  HINTS
    ${TRT_ROOT}/include
    ${TRT_ROOT}/include/x86_64-linux-gnu
    /usr/include
    /usr/include/x86_64-linux-gnu
  PATH_SUFFIXES tensorrt
)
find_library(TRT_LIBRARY nvinfer
  HINTS
    ${TRT_ROOT}/lib
    ${TRT_ROOT}/lib64
    ${TRT_ROOT}/lib/x86_64-linux-gnu
    /usr/lib
    /usr/lib/x86_64-linux-gnu
)
find_library(TRT_ONNX_PARSER_LIBRARY nvonnxparser
  HINTS
    ${TRT_ROOT}/lib
    ${TRT_ROOT}/lib64
    ${TRT_ROOT}/lib/x86_64-linux-gnu
    /usr/lib
    /usr/lib/x86_64-linux-gnu
)

if(TRT_INCLUDE_DIR AND TRT_LIBRARY)
  message(STATUS "TensorRT found: ${TRT_INCLUDE_DIR}")

  add_executable(yoso_infer_trt yoso_infer_trt.cpp)
  target_include_directories(yoso_infer_trt PRIVATE
    ${TRT_INCLUDE_DIR}
    ${CUDAToolkit_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
  )
  target_link_libraries(yoso_infer_trt PRIVATE
    ${TRT_LIBRARY}
    CUDA::cudart
    ${OpenCV_LIBS}
  )

  # yoso_infer_trt does not parse ONNX at runtime. Link parser only if present.
  if(TRT_ONNX_PARSER_LIBRARY)
    target_link_libraries(yoso_infer_trt PRIVATE ${TRT_ONNX_PARSER_LIBRARY})
    message(STATUS "TensorRT ONNX parser found: ${TRT_ONNX_PARSER_LIBRARY}")
  else()
    message(STATUS "TensorRT ONNX parser not found; building TRT runtime target without it")
  endif()

  message(STATUS "[✓] Target 'yoso_infer_trt' configured")
else()
  message(WARNING "TensorRT not found – skipping yoso_infer_trt target")
  message(STATUS "  Set -DTRT_ROOT=/path/to/TensorRT")
  message(STATUS "  TRT_INCLUDE_DIR='${TRT_INCLUDE_DIR}'")
  message(STATUS "  TRT_LIBRARY='${TRT_LIBRARY}'")
  message(STATUS "  TRT_ONNX_PARSER_LIBRARY='${TRT_ONNX_PARSER_LIBRARY}'")
endif()
