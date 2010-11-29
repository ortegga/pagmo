SET(ALIFE_FILES
	__init__.py
	robot.py
	asteroid.py
	task.py
	alifegui.py
	environment.py
	viewer.py
)

SET(ALIFE_MODELS_FILES
	models/asteroid1.blend
	models/asteroid.blend
	models/asteroid_textured.blend
	models/asteroid_texture.jpg
	models/asteroid_textured.x3d
	models/asteroid.x3d
)

SET(ALIFE_UI_FILES
	ui/__init__.py
	ui/alifeui.py
	ui/alifeui.ui
)

INSTALL(FILES ${ALIFE_FILES} DESTINATION ${PYGMO_INSTALL_PATH}/alife)
INSTALL(FILES ${ALIFE_MODELS_FILES} DESTINATION ${PYGMO_INSTALL_PATH}/alife/models)
INSTALL(FILES ${ALIFE_UI_FILES} DESTINATION ${PYGMO_INSTALL_PATH}/alife/ui)