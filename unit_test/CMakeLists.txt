include_directories (${GTEST_INCLUDE_DIRS})

set (CPUTESTS "VectorTest")
set (GPUTESTS "CudaMathTest"
              "GPUArrayDeviceTest")
set (ALLTESTS ${GPUTESTS} ${CPUTESTS})

foreach (UNIT_TEST ${CPUTESTS})
    set (SOURCEFILE ${UNIT_TEST}.cpp)
    add_executable (${UNIT_TEST} ${SOURCEFILE})
endforeach (UNIT_TEST ${CPUTESTS})

foreach (UNIT_TEST ${GPUTESTS})
    set (SOURCEFILE ${UNIT_TEST}.cu)
    cuda_add_executable (${UNIT_TEST} ${SOURCEFILE})
endforeach (UNIT_TEST ${GPUTESTS})

foreach (UNIT_TEST ${ALLTESTS})
    target_link_libraries (
        ${UNIT_TEST} ${MD_ENGINE_LIB_NAME} ${GTEST_BOTH_LIBRARIES} -pthread
        )
    add_test (${UNIT_TEST} ${UNIT_TEST})
endforeach (UNIT_TEST ${ALLTESTS})