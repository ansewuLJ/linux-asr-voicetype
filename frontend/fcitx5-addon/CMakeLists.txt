cmake_minimum_required(VERSION 3.16)
project(voicetype-fcitx5-addon VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Fcitx5Core REQUIRED)
find_package(Fcitx5Config REQUIRED)
find_package(CURL REQUIRED)
find_package(nlohmann_json REQUIRED)

add_library(voicetype-fcitx5 MODULE
    voicetype.cpp
)

if(Fcitx5Core_VERSION VERSION_GREATER_EQUAL "5.1.12")
    target_compile_definitions(voicetype-fcitx5 PRIVATE VOICETYPE_HAVE_ADDON_FACTORY_V2)
endif()

target_link_libraries(voicetype-fcitx5 PRIVATE
    Fcitx5::Core
    Fcitx5::Config
    CURL::libcurl
    nlohmann_json::nlohmann_json
)

set_target_properties(voicetype-fcitx5 PROPERTIES PREFIX "")

install(TARGETS voicetype-fcitx5 DESTINATION "${FCITX_INSTALL_LIBDIR}/fcitx5")

configure_file(voicetype-addon.conf.in voicetype-addon.conf @ONLY)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/voicetype-addon.conf"
        DESTINATION "${FCITX_INSTALL_PKGDATADIR}/addon"
        RENAME voicetype.conf)
