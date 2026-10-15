# --- emlearn_iir stub (added) ---
add_library(usermod_emlearn_iir INTERFACE)

target_sources(usermod_emlearn_iir INTERFACE
    ${CMAKE_CURRENT_LIST_DIR}/../ses_modules/emlearn-micropython/src/emlearn_iir/mod_emlearn_iir.c
)

target_include_directories(usermod_emlearn_iir INTERFACE
    ${CMAKE_CURRENT_LIST_DIR}/../ses_modules/emlearn-micropython/src/emlearn_iir
)

# Link stub into MicroPython's usermod target
target_link_libraries(usermod INTERFACE usermod_emlearn_iir)
# --- end emlearn_iir stub ---
