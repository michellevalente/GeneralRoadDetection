#ifndef ROAD_DETECTOR_H
#define ROAD_DETECTOR_H

#include <string>
#include <numeric>
#include <algorithm>
#include "gaborKernel.h"

class RoadDetector
{
public:
	RoadDetector(std::string dfilename, double dpercImgDetect = 0.5, int dnumSeg = 18, int dnumOrientations = 36, int dnumScales = 6, int dw = 480, int dh = 360);
	~RoadDetector(){};

	Mat image;
	Mat imageGray;
	void initKernel();
	void findVanishingPoint(Point &vanishingPoint);
	void applyFilter();
	void calcOrientationConfiance();
	void drawOrientation(int grid = 15, int lenLine = 10);
	void drawConfidence();
	Point findVanishingPoint(int type = 0, int segNum = 60, int angleNum = 53);
	void calcOrientationConfiance2();
	void findVanishingPoint2();
private:
	float getScoreOrientation(Point p, float angle, int thr);
	float getScore(Point point, float angleStep, float oriNum);
	float getAngle(int x1,int y1,int x2,int y2,float gaborAngle);
	float getScore2(int x,int y,float radius);
	void drawLines(const Point &van_point, const float lineAngle);
	void voter(const Point p, Mat& votes, Mat& dist_table);
	bool liesOnRay(const Point &p, const Point &begin, const Point &end);
	void getDistTable(Mat &dist_table);


	std::string filename;
	int numSeg;
	int numOrientations;
	int numScales;
	int w;
	int h;
	double percImgDetect;
	const int margin_h = 50;
	const int margin_w = 100;

	vector< vector<GaborKernel> > kernels;
	vector<float> orientations;
	vector<Mat> responseOrientation;
	Mat theta, conf, votes;

};

#endif