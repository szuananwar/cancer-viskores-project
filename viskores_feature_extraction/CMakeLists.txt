cmake_minimum_required(VERSION 3.15)
project(ExtractTumorFeatures CXX)

find_package(Viskores REQUIRED)

add_executable(ExtractTumorFeatures ExtractTumorFeatures.cxx)

target_link_libraries(ExtractTumorFeatures PRIVATE
    viskores::cont
    viskores::filter_contour
    viskores::io
)
