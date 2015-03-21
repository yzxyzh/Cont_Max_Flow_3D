cmake_minimum_required(VERSION 2.8)

project(Flow3D)

set(src ContMaxFlow.cpp ContMaxFlow.h main.cpp)

find_package(ITK REQUIRED)

set(armadillo_include /Users/yanzixu/Downloads/armadillo-4.650.4/include)
set(armadillo_lib /Users/yanzixu/Downloads/armadillo-4.650.4/BUILD)
set(intel_cpp_lib /opt/intel/composer_xe_2013_sp1.3.166/compiler/lib /opt/intel/composer_xe_2013_sp1.3.166/mkl/lib)


INCLUDE(${ITK_USE_FILE})

include_directories(${armadillo_include} )
link_directories(${armadillo_lib} ${intel_cpp_lib})

add_executable(Flow3D ${src})

target_link_libraries(Flow3D ${ITK_LIBRARIES} armadillo 	
	libiomp5.a 
	libmkl_intel_thread.a 
	libmkl_intel_lp64.a 
	libmkl_core.a
	libmkl_rt.dylib 

	)