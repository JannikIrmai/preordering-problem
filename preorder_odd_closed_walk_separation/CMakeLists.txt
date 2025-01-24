cmake_minimum_required(VERSION 3.15)
project(preorder_odd_closed_walk_separation)

set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(preorder_odd_closed_walk_separation main.cxx)
install(TARGETS preorder_odd_closed_walk_separation DESTINATION .)
