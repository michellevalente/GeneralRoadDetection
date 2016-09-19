#ifndef ROAD_DETECTOR_PEYMAN_H
#define ROAD_DETECTOR_PEYMAN_H

#include <string>
#include <numeric>
#include <algorithm>
#include "gaborKernel.h"

class RoadDetectorPeyman
{
public:
	RoadDetectorPeyman();
	~RoadDetectorPeyman(){};

	Mat image;
	Mat imageGray;

	void applyFilter(std::string dfilename, int dw, int dh);
	void calcOrientationConfiance();
	Point findVanishingPoint(Point old_vp);
	void drawOrientation(int grid = 15, int lenLine = 8);
	void findRoad();

private:
	void voter(const Point p, Mat& votes, Mat& dist_table, Point old_vp);
	float voteRoad(float angle, float thr);
	Point computeBottomPoint(Point point,float angle);
	std::string filename;
	int numOrientations;
	int numScales;
	int w;
	int h;
	int margin_h;
	int margin_w;
	Point vp;

	vector< vector<GaborKernel> > kernels;
	vector<float> orientations;
	vector<Mat> responseOrientation;
	Mat theta, conf, votes;

};

#endif