find_program (PYTHON_BIN "python")

if (PYTHON_BIN)
    set (SETUP_PY_IN "${CMAKE_CURRENT_SOURCE_DIR}/setup.py.in")
    set (SETUP_PY "${CMAKE_CURRENT_BINARY_DIR}/setup.py")
    set (OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/timestamp")

    get_filename_component (SIM_SOURCES Sim.cpp ABSOLUTE)

    configure_file (${SETUP_PY_IN} ${SETUP_PY})

    add_custom_command (OUTPUT ${OUTPUT}
        COMMAND ${PYTHON_BIN} ${SETUP_PY} build
                        COMMAND ${CMAKE_COMMAND} -E touch ${OUTPUT})

    add_custom_target (PythonLib ALL DEPENDS ${OUTPUT})

    install (CODE "execute_process(COMMAND ${PYTHON_BIN} ${SETUP_PY} install)")
else (PYTHON_BIN)
    message (WARNING "Python binary not found.")
endif (PYTHON_BIN)
