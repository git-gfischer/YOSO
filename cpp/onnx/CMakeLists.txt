if(NOT DEFINED ORT_ROOT)
  set(ORT_ROOT "/usr/local/onnxruntime" CACHE PATH "ONNX Runtime root directory")
endif()

find_path(ORT_INCLUDE_DIR onnxruntime_cxx_api.h
  HINTS ${ORT_ROOT}/include
  PATH_SUFFIXES onnxruntime
  NO_DEFAULT_PATH
)
if(NOT ORT_INCLUDE_DIR)
  find_path(ORT_INCLUDE_DIR onnxruntime_cxx_api.h
    HINTS /usr/local/include /usr/include
    PATH_SUFFIXES onnxruntime
  )
endif()

find_library(ORT_LIBRARY onnxruntime
  HINTS ${ORT_ROOT}/lib
  NO_DEFAULT_PATH
)
if(NOT ORT_LIBRARY)
  find_library(ORT_LIBRARY onnxruntime
    HINTS /usr/local/lib /usr/lib /usr/lib/x86_64-linux-gnu
  )
endif()

if(ORT_INCLUDE_DIR AND ORT_LIBRARY)
  message(STATUS "ONNX Runtime found: ${ORT_INCLUDE_DIR}")

  add_executable(yoso_infer_ort yoso_infer_ort.cpp)
  target_include_directories(yoso_infer_ort PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/../common
    ${ORT_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
  )
  target_link_libraries(yoso_infer_ort PRIVATE
    ${ORT_LIBRARY}
    ${OpenCV_LIBS}
  )
  get_filename_component(ORT_LIBRARY_DIR "${ORT_LIBRARY}" DIRECTORY)
  set_target_properties(yoso_infer_ort PROPERTIES
    BUILD_RPATH "${ORT_LIBRARY_DIR};${ORT_ROOT}/lib"
    INSTALL_RPATH "${ORT_LIBRARY_DIR};${ORT_ROOT}/lib"
  )

  message(STATUS "[✓] Target 'yoso_infer_ort' configured")
else()
  message(WARNING "ONNX Runtime not found – skipping yoso_infer_ort target")
  message(STATUS "  Set -DORT_ROOT=/path/to/onnxruntime")
  message(STATUS "  Download: https://github.com/microsoft/onnxruntime/releases")
endif()
