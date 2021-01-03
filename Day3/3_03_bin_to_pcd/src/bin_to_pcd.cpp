#include <boost/program_options.hpp>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/common/point_operators.h>
#include <pcl/common/io.h>
#include <pcl/search/organized.h>
#include <pcl/search/octree.h>
#include <pcl/search/kdtree.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/filters/conditional_removal.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/surface/gp3.h>
#include <pcl/io/vtk_io.h>
#include <pcl/filters/voxel_grid.h>
 
#include <iostream>
#include <fstream>
 
using namespace pcl;
using namespace std;

int j=0;
string pcd = ".pcd";
string bin = ".bin";
string load(int);


int main(int argc, char **argv){

	string infile;
	string outfile;
	cout << j << endl;

	for (j=0; j<433; j++){

		stringstream ssInt_2;
		ssInt_2 << j;
		infile = load(j);

		fstream input(infile.c_str(), ios::in | ios::binary);
		if(!input.good()){
			cerr << "Could not read file: " << infile << endl;
			exit(EXIT_FAILURE);
		}
		input.seekg(0, ios::beg);

		pcl::PointCloud<PointXYZI>::Ptr points (new pcl::PointCloud<PointXYZI>);
		int i;
		for (i=0; input.good() && !input.eof(); i++) {
			PointXYZI point;
			input.read((char *) &point.x, 3*sizeof(float));
			input.read((char *) &point.intensity, sizeof(float));
			points->push_back(point);
		}
		input.close();
	
		outfile = "/home/user/catkin_ws/src/bin_to_pcd/Data_pcd/pcd_";
		outfile += ssInt_2.str();
		outfile += pcd;

		cout << "Read KTTI point cloud with " << i << " points, writing to " << outfile << endl;
	 
		pcl::PCDWriter writer;

		writer.write<PointXYZI> (outfile, *points, false);
	}
}

string load(int k){
	string infile_name_temp;
	stringstream ssInt;
	ssInt << k;

	if (k<10){
		infile_name_temp = "/home/user/catkin_ws/src/bin_to_pcd/Data_bin/000000000";
		infile_name_temp += ssInt.str();
		infile_name_temp += bin;
	}
	else if (k<100){
		infile_name_temp = "/home/user/catkin_ws/src/bin_to_pcd/Data_bin/00000000";
		infile_name_temp += ssInt.str();
		infile_name_temp += bin;
	}
	else if (k<1000){
		infile_name_temp = "/home/user/catkin_ws/src/bin_to_pcd/Data_bin/0000000";
		infile_name_temp += ssInt.str();
		infile_name_temp += bin;
	}

	return string(infile_name_temp);

}

