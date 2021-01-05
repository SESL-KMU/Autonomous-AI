#최소 요구 버젼 
CMAKE_MINIMUM_REQUIRED (VERSION 2.8 FATAL_ERROR) 

# 패키지 이름 
PROJECT (PCL_TEST)          
MESSAGE (STATUS ${CMAKE_PROJECT_NAME}) #( [<Type>] <메시지> )  


SET(COMPILE_FLAGS "-std=c++11") 

# 의존성 패키지
FIND_PACKAGE (PCL 1.7 REQUIRED)  # 프로그램 실행시 필요한 패키지

# 헤더 디렉토리 지정 (-I)
INCLUDE_DIRECTORIES(
    ${PCL_INCLUDE_DIRS}
    )

# 라이브러리 디렉토리 지정 (-L)   
LINK_DIRECTORIES(${PCL_LIBRARY_DIRS})


# 전처리기 매크로 추가 (-D)
ADD_DEFINITIONS(${COMPILE_FLAGS})
ADD_DEFINITIONS(${PCL_DEFINITIONS})


# 생성할 실행 파일 옵션 
ADD_EXECUTABLE (pcl_test src/pcl_test.cpp) # (<생성될 실행 파일명>   <생성시 사용할 소스코드> )

# Target 링크 옵션 및 라이브러리 지정 (-l)
TARGET_LINK_LIBRARIES(pcl_test ${PCL_LIBRARIES})    #( <Target_이름> <라이브러리> <라이브러리> ... )
                                                    # 실행 파일생성하기에 앞서 링크 해야 하는 라이브러리와 실행 파일 링크
