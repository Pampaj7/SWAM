#include <Python.h>
#include <iostream>

void add_to_sys_path_py(const char *path) {
  PyObject *sysPath = PySys_GetObject("path"); // Get the sys.path object
  PyObject *pathStr =
      PyUnicode_FromString(path); // Convert the path to a Python string
  if (PyList_Append(sysPath, pathStr) != 0) {
    PyErr_Print();
    std::cerr << "Failed to add path to sys.path" << std::endl;
  }
  Py_DECREF(pathStr);
}
void CallPython(const char *module_name, const char *class_name,
                const char *func_name, PyObject *args) {
  add_to_sys_path_py(".");
  add_to_sys_path_py("/Users/pampaj/anaconda3/envs/cpp/lib/python3.12/site-packages/");
  // Import the Python module
  PyObject *pName = PyUnicode_DecodeFSDefault(module_name);
  PyObject *pModule = PyImport_Import(pName);
  Py_DECREF(pName);

  if (pModule != NULL) {
    // Get the class from the module
    PyObject *pClass = PyObject_GetAttrString(pModule, class_name);

    if (pClass && PyCallable_Check(pClass)) {
      // Get the function from the class
      PyObject *pFunc = PyObject_GetAttrString(pClass, func_name);

      if (pFunc && PyCallable_Check(pFunc)) {
        // Call the function with arguments
        PyObject *pValue = PyObject_CallObject(pFunc, args);
        if (pValue != NULL) {
          // Print the result
          PyObject *pStr = PyObject_Str(pValue);
          std::cout << func_name << " returned: " << PyUnicode_AsUTF8(pStr)
                    << std::endl;
          Py_DECREF(pStr);
          Py_DECREF(pValue);
        } else {
          PyErr_Print();
          std::cerr << "Function call failed" << std::endl;
        }
        Py_DECREF(pFunc);
      } else {
        if (PyErr_Occurred())
          PyErr_Print();
        std::cerr << "Cannot find function " << func_name << std::endl;
      }
      Py_DECREF(pClass);
    } else {
      if (PyErr_Occurred())
        PyErr_Print();
      std::cerr << "Cannot find class " << class_name << std::endl;
    }
    Py_DECREF(pModule);
  } else {
    PyErr_Print();
    std::cerr << "Failed to load module " << module_name << std::endl;
  }
}
