#ifndef ROAD_DETECTOR_PEYMAN_H
#define ROAD_DETECTOR_PEYMAN_H

#include <string>
#include <numeric>
#include <algorithm>
#include <queue> 
#include "gaborKernel.h"

class RoadDetectorPeyman
{
public:
	RoadDetectorPeyman();
	~RoadDetectorPeyman(){};

	Mat image;
	Mat imageGray;
	Mat imageHSV;
	Mat channelHSV[3];
	Mat outimage;

	/* Convolves input frame with the Gabor filters at each pixel
	   and calculates the Gabor Energy responses */
	void applyFilter(std::string dfilename, int dw, int dh);
	void applyFilter(Mat file, int dw, int dh);

	/* Suppresses each Gabor energy response by its ortogonal oriented direction
	   and estimate the local dominant orientation at each pixel */
	void calcOrientationConfiance();

	/* Defines a ray for each pixel orientation and votes for the 
	   vanishing point location */
	Point findVanishingPoint(Point old_vp);

	/* Draws the orientation direction for each pixel */
	void drawOrientation(int grid = 15, int lenLine = 8);

	/* Road detection algorithm. Finds the most dominant edge and the second most 
	   dominant edge with a minimum angle between the second and the first one */ 
	Point findRoad();

	/* Draws an arrow from the middle of the image to the vanishing point */
	void direction(Point van_point);

	void roadDetection(float T);
	void drawConfidence();

private:
	/* Algorithm to vote for the vanishing point from on point to all the others in the ray. */
	void voter(const Point p, Mat& votes, Mat& dist_table, Point old_vp);

	/* OCR calculation to vote for the most dominant edge */ 
	float voteRoad(float angle, float thr, Point p);
	float computeOCR(vector<float>& voteEdges, Point point,int initialAngle , float& sumOCR, float& median);

	void regionGrow(Point seed,int t0, int t1, int t2, Mat img, Mat region, int useOrientation);
	
	bool pointIn(Point p);

	float diffPixels(Point p, Point q, Mat img, int t0, int t1, int t2);

	float directionVp(Point p);

	void findLimits();

	void findThr(Mat img, int& t0, int& t1, int& t2);

	void equalizeRegion(Point p1, Point p2);

	void drawEdge(float angle, Scalar color);

	bool isShadow(Point p, Mat img);

	void testShadow(Mat img);

	Vec3b findMean(vector<Point> R, Mat img);
	void regionGrow2(vector<Point> seeds, int t0, int t1, int t2, Mat img, Mat region);

	std::string filename;
	int numOrientations;
	int numScales;
	int w;
	int h;
	int margin_h;
	int margin_w;
	Point vp;
	float perc_look_old;

	float angle_left;
	float angle_right;
	Point triang1,triang2;

	float medium_bright; 

	vector< vector<GaborKernel> > kernels;
	vector<float> orientations;
	vector<Mat> responseOrientation;
	Mat theta, conf, votes;

};

#endif