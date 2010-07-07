ADD_LIBRARY(_keplerian_toolbox SHARED keplerian_toolbox.cpp)
PYTHON_CPPMODULE(_keplerian_toolbox)
INSTALL(FILES __init__.py DESTINATION ${PYGMO_INSTALL_PATH}/keplerian_toolbox)
