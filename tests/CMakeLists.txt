
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

ADD_EXECUTABLE(local_torture_test local_torture_test.cpp)
TARGET_LINK_LIBRARIES(local_torture_test ${MANDATORY_LIBRARIES} pagmo_static)
ADD_TEST(local_torture_test local_torture_test)
IF(ENABLE_MPI)
	ADD_EXECUTABLE(mpi_torture_test mpi_torture_test.cpp)
	TARGET_LINK_LIBRARIES(mpi_torture_test ${MANDATORY_LIBRARIES} pagmo_static)
	ADD_TEST(mpi_torture_test_01 ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 10 ${MPIEXEC_PREFLAGS} mpi_torture_test
		${MPIEXEC_POSTFLAGS})
	ADD_TEST(mpi_torture_test_02 ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS} mpi_torture_test
		${MPIEXEC_POSTFLAGS})
ENDIF(ENABLE_MPI)
