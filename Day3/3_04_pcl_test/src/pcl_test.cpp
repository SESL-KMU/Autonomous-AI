#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>  //`pcl::PointXYZ`을 포함한 포인트 type 구조체 정의 

int main (int argc, char** argv)
{
	pcl::PointCloud<pcl::PointXYZ> cloud; //생성할 PointCloud structure구조체(x,y,z) 정의 

	// 포인트클라우드의 파라미터 설정 : width, height, is_dense
	cloud.width    = ;
	cloud.height   = ;
	cloud.is_dense = ;
	cloud.points.resize (cloud.width * cloud.height);

	//랜덤한 위치 정보를 각 포인트에 지정  
	for (std::size_t i = 0; i < cloud.points.size (); ++i)
	{

	}

	//test_pcd_1.pcd이름으로 저장 
	pcl::io::savePCDFileASCII ("/home/user/pcl_test/src/test_pcd_1.pcd", cloud);

	//정보 출력 
	std::cerr << "Saved " << cloud.points.size () << " data points to test_pcd_1.pcd." << std::endl;

	for (std::size_t i = 0; i < cloud.points.size (); ++i)
		std::cerr << "    " << cloud.points[i].x << " " << cloud.points[i].y << " " << cloud.points[i].z << std::endl;

	return (0);
}

