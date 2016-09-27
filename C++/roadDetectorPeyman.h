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
	void findRoad();

	/* Road detection algorithm. Finds the most dominant edge, constructs a set of line segments
	   for each samples pixel in the edge, compute the OCR for each line and select the one with the highest
	   OCR. Recalculates the vanishing point location from the new line. */
	void findRoad2();

	/* Draws an arrow from the middle of the image to the vanishing point */
	void direction(Point van_point);

	/* Algorithm to detect the sky in the images */
	void detectSky();

private:
	/* Algorithm to vote for the vanishing point from on point to all the others in the ray. */
	void voter(const Point p, Mat& votes, Mat& dist_table, Point old_vp);

	/* OCR calculation to vote for the most dominant edge */ 
	float voteRoad(float angle, float thr, Point p);
	float computeOCR(vector<float>& voteEdges, Point point,int initialAngle , int& sumOCR);

	
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