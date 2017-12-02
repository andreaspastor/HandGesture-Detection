from cx_Freeze import setup, Executable
import os

os.environ['TCL_LIBRARY'] = "C:\\Python36\\tcl\\tcl8.6"
os.environ['TK_LIBRARY'] = "C:\\Python36\\tcl\\tk8.6"

build_exe_options = {"packages": ["tensorflow"], "excludes": ["tkinter"]}
# On appelle la fonction setup
setup(
    name = "HandGestureDetector",
    version = "1.0.0",
    description = "Detection of ten gestures",
    executables = [Executable("testModel.py")],
)