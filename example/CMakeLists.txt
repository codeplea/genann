get_filename_component(LIBRARY_NAME "${CMAKE_CURRENT_SOURCE_DIR}" NAME)
string(REPLACE " " "_" LIBRARY_NAME "${LIBRARY_NAME}")

file(
        COPY "iris.data" "iris.names" "xor.ann"
        DESTINATION "${CMAKE_ARCHIVE_OUTPUT_DIRECTORY}/example"
)

foreach(example RANGE 1 4)
  set(EXEC_NAME "example${example}")

  set(Source_Files "../${EXEC_NAME}.c")
  source_group("Source Files" FILES "${Source_Files}")

  set(TARGET_NAME "${PROJECT_NAME}_${EXEC_NAME}")
  add_executable("${TARGET_NAME}" "${Source_Files}")

  target_include_directories(
          "${TARGET_NAME}"
          INTERFACE
          "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>"
          "$<INSTALL_INTERFACE:include>"
  )

  target_link_libraries(
          "${TARGET_NAME}"
          INTERFACE
          "${PROJECT_LOWER_NAME}_compiler_flags"
  )
  target_link_libraries("${TARGET_NAME}" PRIVATE "genann")

  set_target_properties(
          "${TARGET_NAME}"
          PROPERTIES
          LINKER_LANGUAGE
          C
  )

  # install rules
  set(installable_libs "${TARGET_NAME}" "${PROJECT_LOWER_NAME}_compiler_flags")
  if (TARGET "${DEPENDANT_LIBRARY}")
    list(APPEND installable_libs "${DEPENDANT_LIBRARY}")
  endif ()
  install(TARGETS ${installable_libs}
          DESTINATION "bin/"
          EXPORT "${TARGET_NAME}Targets")

endforeach ()
