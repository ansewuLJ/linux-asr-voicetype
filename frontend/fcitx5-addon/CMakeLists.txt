cmake_minimum_required(VERSION 3.16)
project(qfinput-fcitx5-addon VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Fcitx5Core REQUIRED)
find_package(CURL REQUIRED)
find_package(nlohmann_json REQUIRED)

add_library(qfinput-fcitx5 MODULE
    qfinput.cpp
)

if(Fcitx5Core_VERSION VERSION_GREATER_EQUAL "5.1.12")
    target_compile_definitions(qfinput-fcitx5 PRIVATE QFINPUT_HAVE_ADDON_FACTORY_V2)
endif()

target_link_libraries(qfinput-fcitx5 PRIVATE
    Fcitx5::Core
    CURL::libcurl
    nlohmann_json::nlohmann_json
)

set_target_properties(qfinput-fcitx5 PROPERTIES PREFIX "")

install(TARGETS qfinput-fcitx5 DESTINATION "${FCITX_INSTALL_LIBDIR}/fcitx5")

configure_file(qfinput-addon.conf.in qfinput-addon.conf @ONLY)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/qfinput-addon.conf"
        DESTINATION "${FCITX_INSTALL_PKGDATADIR}/addon"
        RENAME qfinput.conf)
