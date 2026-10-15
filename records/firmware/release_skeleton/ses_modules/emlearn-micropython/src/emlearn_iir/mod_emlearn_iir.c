#include "py/obj.h"
#include "py/runtime.h"
#include "py/objmodule.h"

STATIC const mp_rom_map_elem_t emlearn_iir_module_globals_table[] = {
    { MP_ROM_QSTR(MP_QSTR___name__), MP_ROM_QSTR(MP_QSTR_emlearn_iir) },
    // TODO: add real functions here in the future.
};

STATIC MP_DEFINE_CONST_DICT(emlearn_iir_module_globals, emlearn_iir_module_globals_table);

// This is the symbol the linker wants.
const mp_obj_module_t mp_module_emlearn_iir = {
    .base = { &mp_type_module },
    .globals = (mp_obj_dict_t*)&emlearn_iir_module_globals,
};

// Register module in MicroPython (2-argument form for IDF4/MicroPython v1.20)
MP_REGISTER_MODULE(MP_QSTR_emlearn_iir, mp_module_emlearn_iir);
