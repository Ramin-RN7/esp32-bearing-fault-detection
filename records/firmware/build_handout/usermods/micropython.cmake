add_library(usermod_emlearn_iir INTERFACE)
target_sources(usermod_emlearn_iir INTERFACE
    ${CMAKE_CURRENT_LIST_DIR}/../ses_modules/emlearn-micropython/src/emlearn_iir/mod_emlearn_iir.c
)
target_include_directories(usermod_emlearn_iir INTERFACE
    ${CMAKE_CURRENT_LIST_DIR}/../ses_modules/emlearn-micropython/src/emlearn_iir
)
target_link_libraries(usermod INTERFACE usermod_emlearn_iir)
