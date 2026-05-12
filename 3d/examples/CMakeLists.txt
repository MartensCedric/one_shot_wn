add_executable(mesh_demo            mesh_demo.cpp)
add_executable(parametric_demo      parametric_demo.cpp)
add_executable(bezier_patch_demo    bezier_patch_demo.cpp)
add_executable(compare_with_libigl  compare_with_libigl.cpp)

foreach(tgt mesh_demo parametric_demo bezier_patch_demo compare_with_libigl)
  target_link_libraries(${tgt} PRIVATE gwn3d)
endforeach()
