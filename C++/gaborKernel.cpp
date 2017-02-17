#include "gaborKernel.h"

GaborKernel::GaborKernel(){}

GaborKernel::~GaborKernel(){}

GaborKernel::GaborKernel(float dorientation, int dscale, double dSigma, double dF)
{
	Sigma = dSigma;
	orientation = dorientation;
	scale = dscale;
	F = dF;
	K = ( CV_PI / 2.0 ) / (pow(F, float(scale)));
	Width = cvRound(Sigma / K  * 6 + 1);

	if((Width % 2) == 0)
		Width = Width + 1;

	createKernel();
} 

void GaborKernel::init(float Sigma, float F, float dorientation, int dscale)
{
	orientation = dorientation;
	scale = dscale;

	K = ( CV_PI / 2.0 ) / (pow(F, float(scale)));

	Width = cvRound(Sigma / K  * 6 + 1);

	if((Width % 2) == 0)
		Width = Width + 1;

	createKernel();
}

void GaborKernel::createKernel()
{
	Real.create(Width, Width, CV_32FC1);
	Imag.create(Width, Width, CV_32FC1);
	int x,y;
	double k_2 = K * K;
	Sigma = 2 * CV_PI;
	double sigma_2 = Sigma * Sigma;
	double cosOrientation = K * cos(orientation);
	double sinOrientation = K * sin(orientation);

	for(int i = 0; i < Width; i++)
	{
		for(int j = 0; j < Width; j++)
		{
			x = i - (Width - 1) / 2;
			y = j - (Width - 1) / 2;
			double val1 = (k_2 / sigma_2) * exp(- (x*x + y*y) * k_2 / (2*sigma_2));
			double val2 = cos(cosOrientation * x + sinOrientation * y) - exp(-(sigma_2 / 2));
			double val3 = sin(cosOrientation * x + sinOrientation * y);
			Real.at<float>(i, j) = val1 * val2;
			Imag.at<float>(i, j) = val1 * val3;
		}
	}
}

void GaborKernel::applyKernel(Mat &imgSrc, Mat &imgDst)
{
	Mat mat = imgSrc.clone();

    Mat rmat(imgSrc.rows, imgSrc.cols, CV_32FC1);
    Mat imat(imgSrc.rows, imgSrc.cols, CV_32FC1);
    Mat summat(imgSrc.rows, imgSrc.cols, CV_32FC1);

	Mat resultIm(imgSrc.rows, imgSrc.cols, CV_32FC1);
	Mat resultReal(imgSrc.rows, imgSrc.cols, CV_32FC1);

	filter2D(mat, rmat, CV_32F, Real, Point((Width-1)/2, (Width-1)/2));
    filter2D(mat, imat, CV_32F, Imag, Point((Width-1)/2, (Width-1)/2));

    pow(rmat, 2, rmat);
    pow(imat, 2, imat);

    sqrt(rmat + imat, summat);
    summat.copyTo(imgDst);
    
}

