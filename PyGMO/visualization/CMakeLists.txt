# We'll need FTGL
#FIND_LIBRARY(FTGL_LIBRARY NAMES ftgl)
#FIND_PATH(FTGL_INCLUDE_DIR NAMES FTGL/ftgl.h)
#IF(NOT FTGL_LIBRARY OR NOT FTGL_INCLUDE_DIR)
#   MESSAGE(FATAL_ERROR "Visualization module was requested, but the library FTGL could not be located. Please make sure that you have the FTGL library installed")
#ENDIF(NOT FTGL_LIBRARY OR NOT FTGL_INCLUDE_DIR)
#MESSAGE(STATUS "FTGL library: ${FTGL_LIBRARY}")
#MESSAGE(STATUS "FTGL include dir: ${FTGL_INCLUDE_DIR}")

# Try to import pythong-ftgl
#EXECUTE_PROCESS(COMMAND ${PYTHON_EXECUTABLE} -c "import FTGL" FTGL_IMPORT_OUTPUT)
#IF(FTGL_IMPORT_OUPTUT)
#   MESSAGE(FATAL_ERROR "Visualization module was requested, but the python module 'FTGL' could not be located.")
#ENDIF(FTGL_IMPORT_OUPTUT)

# Install
INSTALL(FILES __init__.py frange.py traj3d.py Vera.ttf DESTINATION ${PYGMO_INSTALL_PATH}/visualization)
