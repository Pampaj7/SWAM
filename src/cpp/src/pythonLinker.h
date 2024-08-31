// pythonLinker.h
#include <Python.h>
#ifndef PYTHON_LINKER_H
#define PYTHON_LINKER_H

void CallPython(const char *module_name, const char *class_name,
                const char *function_name, PyObject *args);
void add_to_sys_path_py(const char *path);

#endif // PYTHON_LINKER_H
