ADD_EXECUTABLE(test_001 test_001.cpp)
TARGET_LINK_LIBRARIES(test_001 ${MANDATORY_LIBRARIES} pagmo_static)
ADD_TEST(test_001 test_001)


IF(BUILD_CUDA_TEST)
  IF (ENABLE_CUDA)
    # use nvcc to build the main app
    ADD_EXECUTABLE(cuodetest integrators/odetest.cpp)
    ADD_EXECUTABLE(cunnettest ann_toolbox/nnettest.cpp)
    ADD_EXECUTABLE(cufittest fitness/fittest.cpp)
    TARGET_LINK_LIBRARIES(cuodetest pagmo_static gpgpulib ${MANDATORY_LIBRARIES})
    TARGET_LINK_LIBRARIES(cunnettest pagmo_static gpgpulib ${MANDATORY_LIBRARIES})
    TARGET_LINK_LIBRARIES(cufittest pagmo_static gpgpulib ${MANDATORY_LIBRARIES})
  ELSE (ENABLE_CUDA)
    #ADD_EXECUTABLE(cuda_test cuda_test.cu)
  ENDIF(ENABLE_CUDA)
ENDIF(BUILD_CUDA_TEST)
