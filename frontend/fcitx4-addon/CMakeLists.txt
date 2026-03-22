cmake_minimum_required(VERSION 3.16)
project(voicetype-fcitx4-addon VERSION 0.1.0 LANGUAGES C)

include(GNUInstallDirs)
find_package(PkgConfig REQUIRED)
pkg_check_modules(FCITX REQUIRED fcitx)

add_library(fcitx-voicetype MODULE voicetype_fcitx4.c)
target_include_directories(fcitx-voicetype PRIVATE ${FCITX_INCLUDE_DIRS})
target_compile_options(fcitx-voicetype PRIVATE ${FCITX_CFLAGS_OTHER})
target_link_directories(fcitx-voicetype PRIVATE ${FCITX_LIBRARY_DIRS})
target_link_libraries(fcitx-voicetype PRIVATE ${FCITX_LIBRARIES})
set_target_properties(fcitx-voicetype PROPERTIES PREFIX "")

install(TARGETS fcitx-voicetype DESTINATION "${CMAKE_INSTALL_LIBDIR}/fcitx")
install(FILES fcitx-voicetype.conf DESTINATION "${CMAKE_INSTALL_DATADIR}/fcitx/addon")
install(FILES fcitx-voicetype.desc DESTINATION "${CMAKE_INSTALL_DATADIR}/fcitx/configdesc")
install(FILES fcitx-voicetype.config DESTINATION "${CMAKE_INSTALL_DATADIR}/fcitx/conf")
