#ifndef GABOR_KERNEL_H
#define GABOR_KERNEL_H

#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;

class GaborKernel
{
public:
	GaborKernel();
	~GaborKernel();
	GaborKernel(float orientation, int scale, double Sigma, double F);
	void createKernel();
	void init(float Sigma, float F, float orientation, int numScales);
	void applyKernel(Mat &imgSrc, Mat &imgDst);

protected:
	int Width;
	float orientation;
	int scale;
	double Sigma;
	double F;
	Mat Imag;
	Mat Real;
	double K;
};

#endif