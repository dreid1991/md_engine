include_directories (${GTEST_INCLUDE_DIRS})
add_executable (runVectorTest VectorTest.cpp)

target_link_libraries(
    runVectorTest ${MD_ENGINE_LIB_NAME} ${GTEST_BOTH_LIBRARIES} -pthread
    )

add_test (VectorTest runVectorTest)