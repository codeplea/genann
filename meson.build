project('genann', 'c', license : 'zlib')

genann_lib = static_library('genann',
    files('genann.c'),
    include_directories : '.')

genann_dep = declare_dependency(
    link_with : genann_lib,
    version : meson.project_version(),
    include_directories : '.')
