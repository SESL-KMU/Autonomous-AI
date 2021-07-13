#include <iostream>
#include <string>
#include <stdio.h>

#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/conversions.h>

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/features/normal_3d.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/features/moment_of_inertia_estimation.h>

#include <Eigen/Dense>

typedef pcl::PointCloud<pcl::PointXYZ> PointCloud;
using namespace std;
int j = 0;

PointCloud::Ptr Message(PointCloud::Ptr);
PointCloud::Ptr RoI_Filtering(PointCloud::Ptr);
PointCloud::Ptr RANSAC(PointCloud::Ptr);
pcl::PointCloud<pcl::PointXYZI> Clustering(PointCloud::Ptr);
visualization_msgs::Marker Box(PointCloud::Ptr);
visualization_msgs::MarkerArray msg_marker;


int main(int argc, char** argv) {

	char PcdFileName[1000];

	ros::init(argc, argv, "lidar_data_processing_node");

	ros::NodeHandle nh;
	ros::Publisher pub = nh.advertise<PointCloud>("pointcloud", 1);  					//원본 포인트 클라우드 publisher 생성
	ros::Publisher pub_filter = nh.advertise<PointCloud>("pointcloud_filter", 1);  			//ROI 적용 클라우드 publisher 생성
	ros::Publisher pub_RANSAC = nh.advertise<PointCloud>("pointcloud_RANSAC", 1);  			//RANSAC 적용 클라우드 publisher 생성
	ros::Publisher pub_ec = nh.advertise<sensor_msgs::PointCloud2>("pointcloud_ec", 1);  		//EC 적용 클라우드 publisher 생성
	ros::Publisher pub_marker = nh.advertise<visualization_msgs::MarkerArray>("msg_marker", 100);//Detection Box 적용 Marker Array publisher 생성

	ros::Rate loop_rate(10);


	while (nh.ok())
	{
		msg_marker.markers.clear();
		PointCloud::Ptr msg(new PointCloud);			//rviz에 전달되는 메세지 초기화
		PointCloud::Ptr msg_filter(new PointCloud);  		//rviz에 전달되는 filter 메세지 초기화
		PointCloud::Ptr msg_RANSAC(new PointCloud);  	//rviz에 전달되는 RANSAC 메세지 초기화
		PointCloud::Ptr msg_ec(new PointCloud);  		//rviz에 전달되는 EC 메세지 초기화

		PointCloud::Ptr cloud(new PointCloud);  		// 원본 클라우드 데이터
		PointCloud::Ptr cloud_filter(new PointCloud);  		// ROI 적용 클라우드 데이터
		PointCloud::Ptr cloud_RANSAC(new PointCloud);  	//지면 제거 클라우드 데이터
		pcl::PointCloud<pcl::PointXYZI> cloud_ec;  		//군집화 클라우드 데이터

/********************************************main********************************************/

		sprintf(PcdFileName, "C:/catkin_ws/src/lidar_data_processing/Data_pcd/pcd_%d.pcd", j++);
		pcl::io::loadPCDFile<pcl::PointXYZ>(PcdFileName, *cloud);  //pcd파일이 변경될 때마다 새로운 cloud 생성
		cout << j << endl;

		//cloud_filter = RoI_Filtering(cloud);
		//cloud_RANSAC = RANSAC(cloud_filter);
		//cloud_ec = Clustering(cloud_RANSAC);

		//msg = Message(cloud);				//원본 클라우드 데이터 msg
		//msg_filter = Message(cloud_filter);		//필터링 결과 클라우드 데이터 msg
		//msg_RANSAC = Message(cloud_RANSAC);		//RANSAC 알고리즘 결과 클라우드 데이터 msg

		//pcl::PCLPointCloud2 cloud_p;
		//pcl::toPCLPointCloud2(cloud_ec, cloud_p);
		//sensor_msgs::PointCloud2 output;
		//pcl_conversions::fromPCL(cloud_p, output);
		//output.header.frame_id = "map";			//Clustering 결과 msg (ROS Message Type으로 변환)

		//pub.publish (msg);
		//pub_filter.publish (msg_filter);
		//pub_RANSAC.publish (msg_RANSAC);
		//pub_ec.publish (output);
		//pub_marker.publish(msg_marker);			//각 msg publish


/********************************************************************************************/
		msg_marker.markers.clear();
		ros::spinOnce();
		loop_rate.sleep();

		if (j>80) j = 0;

	}
}



/*********************************************************Message*********************************************************/

PointCloud::Ptr Message(PointCloud::Ptr cloud) {

	double x_cloud; double y_cloud; double z_cloud;  	//포인트 클라우드 데이터 시각화를 위한 변수 선언
	PointCloud::Ptr msg(new PointCloud);			//메세지 초기화

	msg->header.frame_id = "map";
	msg->height = cloud->height;
	msg->width = cloud->width;

	for (size_t i = 0; i < cloud->size(); i++) {
		x_cloud = cloud->points[i].x;
		y_cloud = cloud->points[i].y;
		z_cloud = cloud->points[i].z;
		msg->points.push_back(pcl::PointXYZ(x_cloud, y_cloud, z_cloud));
	}

	return msg;
};

/******************************************************RoI Filtering******************************************************/

PointCloud::Ptr RoI_Filtering(PointCloud::Ptr cloud) {

	pcl::PassThrough<pcl::PointXYZ> pass;  		//ROI 적용을 위한 객체 생성
	PointCloud::Ptr cloud_filter(new PointCloud);  	// ROI 적용 클라우드 데이터

	pass.setInputCloud(cloud);		//입력 
	pass.setFilterFieldName("z");		//적용할 좌표 축
	pass.setFilterLimits(-3, 5);		//적용할 값 (최소, 최대 값)
	pass.filter(*cloud_filter); 		//필터 적용 

	pass.setInputCloud(cloud_filter);	//입력 
	pass.setFilterFieldName("y");		//적용할 좌표 축
	pass.setFilterLimits(-5, 2.8);		//적용할 값 (최소, 최대 값)
	pass.filter(*cloud_filter);		//필터 적용 

	pass.setInputCloud(cloud_filter);	//입력 
	pass.setFilterFieldName("x");		//적용할 좌표 축
	pass.setFilterLimits(-20, 30);		//적용할 값 (최소, 최대 값)
	pass.filter(*cloud_filter);		//필터 적용 

	return cloud_filter;
};


/******************************************************RANSAC Algorithm******************************************************/

PointCloud::Ptr RANSAC(PointCloud::Ptr cloud) {

	pcl::SACSegmentation<pcl::PointXYZ> seg;  							//RANSAC 적용을 위한 객체 생성
	PointCloud::Ptr cloud_inlier(new PointCloud);  	//지면 정보 클라우드 데이터
	PointCloud::Ptr cloud_inlier_neg(new PointCloud);  	//지면 제거 클라우드 데이터
	pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients());
	pcl::PointIndices::Ptr inliers(new pcl::PointIndices());	// Object for storing the plane model coefficients.


	seg.setOptimizeCoefficients(true);		// Enable model coefficient refinement (optional).
	seg.setInputCloud(cloud);				//입력 
	seg.setModelType(pcl::SACMODEL_PLANE);	//적용 모델 (지면을 제거하기 위한 plane 모델 사용)
	seg.setMethodType(pcl::SAC_RANSAC);		//Method Type RANSAC 사용
	seg.setMaxIterations(300);			//최대 실행 수
	seg.setDistanceThreshold(0.16);		//inlier로 처리할 거리 정보
										
	seg.segment(*inliers, *coefficients);		//세그멘테이션 적용 
	pcl::copyPointCloud<pcl::PointXYZ>(*cloud, *inliers, *cloud_inlier);  //지면정보 추출

	pcl::ExtractIndices<pcl::PointXYZ> extract;
	extract.setInputCloud(cloud);
	extract.setIndices(inliers);
	extract.setNegative(true);//false
	extract.filter(*cloud_inlier_neg);


	return cloud_inlier_neg;
};


/******************************************************Clustering******************************************************/

pcl::PointCloud<pcl::PointXYZI> Clustering(PointCloud::Ptr cloud) {
	// 탐색을 위한 KdTree 오브젝트 생성 //Creating the KdTree object for the search method of the extraction
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
	tree->setInputCloud(cloud);  //KdTree 생성 

	std::vector<pcl::PointIndices> cluster_indices;       // 군집화된 결과물의 Index 저장, 다중 군집화 객체는 cluster_indices[0] 순으로 저장 

														  // 군집화 오브젝트 생성  
	pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
	ec.setInputCloud(cloud);       		// 입력   
	ec.setClusterTolerance(0.7);
	ec.setMinClusterSize(100);    		// 최소 포인트 수 
	ec.setMaxClusterSize(6000);  		// 최대 포인트 수
	ec.setSearchMethod(tree);     		// 위에서 정의한 탐색 방법 지정 
	ec.extract(cluster_indices);  		// 군집화 적용 


	pcl::PointCloud<pcl::PointXYZI> TotalCloud;
	msg_marker.markers.clear();
	int k = 0;

	for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin(); it != cluster_indices.end(); ++it) {

		//PointCloud::Ptr box_points (new PointCloud);
		//visualization_msgs::Marker box_marker;

		for (std::vector<int>::const_iterator pit = it->indices.begin(); pit != it->indices.end(); ++pit) {

			pcl::PointXYZ pt = cloud->points[*pit];
			pcl::PointXYZI pt2;
			pt2.x = pt.x, pt2.y = pt.y, pt2.z = pt.z;
			pt2.intensity = (float)(k + 1);

			//box_points->points.push_back(pcl::PointXYZ(pt2.x, pt2.y, pt2.z));
			TotalCloud.push_back(pt2);
		}

		//box_marker = Box(box_points);
		//box_marker.id = k;
		//msg_marker.markers.push_back(box_marker);
		//box_marker.points.clear();

		k++;
	}

	return TotalCloud;
};


/******************************************************Detection Box******************************************************/

visualization_msgs::Marker Box(PointCloud::Ptr box_points) {
	pcl::MomentOfInertiaEstimation<pcl::PointXYZ> feature_extractor;
	feature_extractor.setInputCloud(box_points);
	feature_extractor.compute();

	std::vector<float> moment_of_inertia;
	std::vector<float> eccentricity;

	pcl::PointXYZ min_point_OBB;
	pcl::PointXYZ max_point_OBB;
	pcl::PointXYZ position_OBB;
	Eigen::Matrix3f rotational_matrix_OBB;
	float major_value, middle_value, minor_value;
	Eigen::Vector3f major_vector, middle_vector, minor_vector;
	Eigen::Vector3f mass_center;

	feature_extractor.getMomentOfInertia(moment_of_inertia);
	feature_extractor.getEccentricity(eccentricity);
	feature_extractor.getOBB(min_point_OBB, max_point_OBB, position_OBB, rotational_matrix_OBB);
	feature_extractor.getEigenValues(major_value, middle_value, minor_value);
	feature_extractor.getEigenVectors(major_vector, middle_vector, minor_vector);
	feature_extractor.getMassCenter(mass_center);

	Eigen::Vector3f position(position_OBB.x, position_OBB.y, position_OBB.z);
	Eigen::Quaternionf quat(rotational_matrix_OBB);

	//visualization marker
	std::string ns;

	visualization_msgs::Marker bbx_marker;
	bbx_marker.header.frame_id = "map";
	bbx_marker.header.stamp = ros::Time::now();
	bbx_marker.ns = ns;

	bbx_marker.type = visualization_msgs::Marker::CUBE;
	bbx_marker.action = visualization_msgs::Marker::ADD;

	bbx_marker.pose.position.x = position_OBB.x;
	bbx_marker.pose.position.y = position_OBB.y;
	bbx_marker.pose.position.z = position_OBB.z;
	bbx_marker.pose.orientation.x = quat.x();
	bbx_marker.pose.orientation.y = quat.y();
	bbx_marker.pose.orientation.z = quat.z();
	bbx_marker.pose.orientation.w = quat.w();

	if ((max_point_OBB.x - min_point_OBB.x < 5) && (max_point_OBB.x - min_point_OBB.x > 1))
		bbx_marker.scale.x = (max_point_OBB.x - min_point_OBB.x);
	bbx_marker.scale.y = (max_point_OBB.y - min_point_OBB.y);
	bbx_marker.scale.z = (max_point_OBB.z - min_point_OBB.z);


	bbx_marker.color.b = 0.0f;
	bbx_marker.color.g = 0.8f;
	bbx_marker.color.r = 0.0f;
	bbx_marker.color.a = 0.3;


	bbx_marker.lifetime = ros::Duration();

	return bbx_marker;
}

