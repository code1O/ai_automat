# ============ UTIL CONSOLE TASK MANAGEMENT ============
#
# ------------ INSTALLATION MANAGEMENT ------------
#
#
# - PYTHON INSTALLATION
#
# - NPM, PNPM INSTALLATION
#
# ======================================================

import sys
import subprocess
from rich.console import Console

_python_version = sys.version[:4]
_short_py_version = _python_version.replace(".", "")

if sys.platform == "win32":
    _PythonPath = "~/AppData/Local/Packages"
    _PythonPath = f"{_PythonPath}/PythonSoftwareFoundation.Python.{_python_version}_qbz5n2kfra8p0"
    _PythonLibs = f"{_PythonPath}/Python{_short_py_version}/site-packages"

console = Console()

class _pip_python:
    def __init__(self, mod_lib: str) -> None:
        self.mod_lib = mod_lib
    
    def install(self, modular: str = "python", force: bool = False) -> None:
        """
        modular: If install with python, npm, yarn, etc...
        
        force: Works better in npm, pnpm or yarn, etc...
        """
        
        _py_command = [f"python{_python_version}", "-m", "pip", "install", self.mod_lib]
        _other_command = [modular, ""]
        
        try:
            with console.status(f"Installing with {modular}", spinner="dots"):
                subprocess.run()
        except FileExistsError(f"{_PythonLibs}/{self.mod_lib}" or f""):
            console.log(f"Error at {modular} installation"), print(f"{self.mod_lib} already exists! try another")
    
    def uninstall(self, modular: str) -> None:
        ...

pip_install = lambda mod_lib, modular: _pip_python(mod_lib).install(modular)

if __name__ == "__main__":
    globals()[sys.argv[1]](*sys.argv[2:])