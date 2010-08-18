

# OpenGL
FIND_PACKAGE(OpenGL)
IF(NOT OPENGL_FOUND)
   MESSAGE(FATAL_ERROR "Library OpenGL not found. Please make sure that you have the OpenGL library installed")
ENDIF(NOT OPENGL_FOUND)

# FTGL
FIND_LIBRARY(FTGL_LIBRARY NAMES ftgl)
FIND_PATH(FTGL_INCLUDE_DIR NAMES FTGL/ftgl.h)
IF(NOT FTGL_LIBRARY OR NOT FTGL_INCLUDE_DIR)
   MESSAGE(FATAL_ERROR "Library FTGL not found. Please make sure that you have the FTGL library installed")
ENDIF(NOT FTGL_LIBRARY OR NOT FTGL_INCLUDE_DIR)
INCLUDE_DIRECTORIES(${FTGL_INCLUDE_DIR})
MESSAGE(STATUS "FTGL library: ${FTGL_LIBRARY}")
MESSAGE(STATUS "FTGL include dir: ${FTGL_INCLUDE_DIR}")

# FreeType
INCLUDE(FindFreetype)
IF(NOT FREETYPE_FOUND)
   MESSAGE(FATAL_ERROR "Library FreeType not found. Please make sure that you have the FreeType library installed")
ENDIF(NOT FREETYPE_FOUND)
INCLUDE_DIRECTORIES(${FREETYPE_INCLUDE_DIRS})
MESSAGE(STATUS "FreeType libraries: ${FREETYPE_LIBRARIES}")
MESSAGE(STATUS "FreeType include dir: ${FREETYPE_INCLUDE_DIRS}")

ADD_LIBRARY(FTGL SHARED ftgl.cpp)

# Here we need to set a proper prefix and suffix for the pyftgl module file: we want to strip the 'lib' prefix (as
# Python modules do not need it) and use as extension either .so (for UNIX-like OS) or .pyd (for Windows). The
# PYDEXTENSION and SOEXTENSION variables are set in PaGMOPythonSetup.cmake.
SET_TARGET_PROPERTIES(FTGL PROPERTIES PREFIX "")
IF(PYDEXTENSION)
	SET_TARGET_PROPERTIES(FTGL PROPERTIES SUFFIX ".pyd")
ELSEIF(SOEXTENSION)
	SET_TARGET_PROPERTIES(FTGL PROPERTIES SUFFIX ".so")
ENDIF(PYDEXTENSION)

# We need to link in all the standard PaGMO libraries, the Python libraries, Freetype and FTGL.
TARGET_LINK_LIBRARIES(FTGL ${MANDATORY_LIBRARIES} ${PYTHON_LIBRARIES} ${FREETYPE_LIBRARIES} ${FTGL_LIBRARY})
INSTALL(TARGETS FTGL
	RUNTIME DESTINATION ${PYGMO_INSTALL_PATH}/visualization
	LIBRARY DESTINATION ${PYGMO_INSTALL_PATH}/visualization
)
