
cmake_minimum_required(VERSION 3.13)                 
project( Tutorial_MASK_RCNN )

find_package( OpenCV 4.0.0 REQUIRED )

file(COPY files DESTINATION data)

add_executable( ${PROJECT_NAME} main.cpp )
target_link_libraries( ${PROJECT_NAME} ${OpenCV_LIBS} )