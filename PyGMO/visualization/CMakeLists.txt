

# Try to import pythong-ftgl
EXECUTE_PROCESS(COMMAND ${PYTHON_EXECUTABLE} -c "import FTGL" RESULT_VARIABLE FTGL_IMPORT_RESULT OUTPUT_QUIET ERROR_QUIET)
IF(FTGL_IMPORT_RESULT)
	MESSAGE(STATUS "pyftgl was not found on your system, using bundled copy.")
	ADD_SUBDIRECTORY(pyftgl)
ENDIF(FTGL_IMPORT_RESULT)

# Install
INSTALL(FILES __init__.py frange.py traj3d.py traj3d_object.py traj3d_traj.py traj3d_path.py traj3d_planet.py FreeSans.ttf DESTINATION ${PYGMO_INSTALL_PATH}/visualization)
