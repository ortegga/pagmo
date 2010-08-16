

# Try to import pythong-ftgl
EXECUTE_PROCESS(COMMAND ${PYTHON_EXECUTABLE} -c "import FTGL" FTGL_IMPORT_OUTPUT)
IF(FTGL_IMPORT_OUPTUT)
   ADD_SUBDIRECTORY(pyftgl)
ENDIF(FTGL_IMPORT_OUPTUT)

# Install
INSTALL(FILES __init__.py frange.py traj3d.py traj3d_object.py traj3d_traj.py traj3d_path.py traj3d_planet.py Vera.ttf DESTINATION ${PYGMO_INSTALL_PATH}/visualization)
