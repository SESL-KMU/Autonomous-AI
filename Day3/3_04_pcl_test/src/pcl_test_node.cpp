#include <iostream>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>  //"pcl::PointXYZ"을 포함한 포인트 type 구조체 정의 

int main (int argc, char** argv)
{
	ros::init (argc, argv, "pcl_test_node");

	ros::NodeHandle nh;
	ros::Publisher publisher = nh.advertise<pcl::PointCloud<pcl::PointXYZ>> ("pointcloud", 1);  				//원본 포인트 클라우드 publisher 생성

	pcl::PointCloud<pcl::PointXYZ> cloud; //생성할 PointCloud structure구조체(x,y,z) 정의 
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_msg (new pcl::PointCloud<pcl::PointXYZ>);
	ros::Rate loop_rate(10);

	// 포인트클라우드의 파라미터 설정 : width, height, is_dense

	cloud.width    = 20;
	cloud.height   = 20;
	cloud.is_dense = true;
	cloud.points.resize (cloud.width * cloud.height);

	cloud_msg->header.frame_id = "map";
	cloud_msg->height = cloud.height;
	cloud_msg->width = cloud.width;

	//랜덤한 위치 정보를 각 포인트에 지정  
	for (std::size_t i = 0; i < cloud.points.size (); ++i)
	{
		cloud.points[i].x = 1024 * rand () / (RAND_MAX + 1.0f);
		cloud.points[i].y = 1024 * rand () / (RAND_MAX + 1.0f);
		cloud.points[i].z = 1024 * rand () / (RAND_MAX + 1.0f);
		cloud_msg->points.push_back (pcl::PointXYZ (cloud.points[i].x, cloud.points[i].y, cloud.points[i].z));
	}

	//test_pcd_1.pcd이름으로 저장 
	pcl::io::savePCDFileASCII ("C:/catkin_ws/src/pcl_test/src/test_pcd_1.pcd", cloud);

	//정보 출력 
	std::cerr << "Saved " << cloud.points.size () << " data points to test_pcd.pcd." << std::endl;

	for (std::size_t i = 0; i < cloud.points.size (); ++i)
		std::cerr << "    " << cloud.points[i].x << " " << cloud.points[i].y << " " << cloud.points[i].z << std::endl;
	
	while (nh.ok()){
		publisher.publish(cloud_msg);
	}
	
	ros::spinOnce ();
	loop_rate.sleep ();	

	return (0);
}

