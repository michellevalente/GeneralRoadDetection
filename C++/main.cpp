#include "gaborKernel.h"
#include "roadDetector.h"
#include "roadDetectorPeyman.h"

int captureFrame(string src, string dst, int sample_interval = 100, 
	int width = 480, int height = 360, int fps = 10)
{
	
	VideoCapture cap(src);

	if(!cap.isOpened())
	{
		cerr << "video not opened!" << endl;
		return 0;
	}

	cap.set(CV_CAP_PROP_FRAME_WIDTH, width);
    cap.set(CV_CAP_PROP_FRAME_HEIGHT, height);
    cap.set(CV_CAP_PROP_FPS, fps);

    int cnt = 0;
	while(true)
	{		
		char filename[512];
		Mat frame;
		for (int i = 0; i < sample_interval; i++)
			cap.grab();
		cap.retrieve(frame);		

		sprintf(filename, "%s%010d.png", dst.c_str(), ++cnt);
		imwrite(string(filename), frame);		
	}

	return cnt;
	
}


void genVideo(string src, int numFrames, string output_name = "output.avi", 
	double fps = 10.0, int wFrame = 640, int hFrame = 480, char * showOrientation = "0", char * showRoad = "0")
{
	//VideoWriter vw(output_name, CV_FOURCC('M','J','P','G'), fps, Size(wFrame, hFrame));
	// if (!vw.isOpened())
	// {
	// 	cout << "video writer not initialized" << endl;
	// 	return;
	// }
	RoadDetectorPeyman roadDetector;
	Point vp = Point(0 , 0);

	for (int i = 1; i < numFrames; i++)
	{		
		char filename[512];

		sprintf(filename, "%s%010d.png", src.c_str(), i);
		cout << endl << filename << endl;

		roadDetector.applyFilter(filename, wFrame , hFrame);
		roadDetector.calcOrientationConfiance();
		if(!strcmp(showOrientation, "1"))
			roadDetector.drawOrientation();

		vp = roadDetector.findVanishingPoint(vp);

		if(!strcmp(showRoad,"1"))
			roadDetector.findRoad();
		
		imshow("", roadDetector.image);
		waitKey(15);
		//vw.write(roadDetector.image);
	}
}
void testImage(string img_dir,int wFrame = 480, int hFrame = 360, char * showOrientation = "0", char * showRoad = "0")
{
	RoadDetectorPeyman roadDetector;
	Point vp = Point(0,0);
	roadDetector.applyFilter(img_dir,wFrame, hFrame);
	roadDetector.calcOrientationConfiance();
	if(!strcmp(showOrientation, "1"))
			roadDetector.drawOrientation();

	vp = roadDetector.findVanishingPoint(vp);

	if(!strcmp(showRoad,"1"))
		roadDetector.findRoad();

	imshow("Final image", roadDetector.image);
	waitKey(0);
}


int main(int argc,char *argv[])
{
	if(argc != 4){
		cout << "Missing arguments!" << endl;
		return 1;
	}

	if(!strcmp(argv[1],"road"))
		genVideo("../images/road/data/", 20, "test_results/test.avi", 10, 960, 290, argv[2], argv[3]);
	else if(!strcmp(argv[1],"dirt"))
		genVideo("../images/frames_dirt/", 110, "test_results/test.avi", 10, 360, 240, argv[2], argv[3]);
	else if(!strcmp(argv[1], "snow"))
		testImage("../images/snow.jpg", 480,360, argv[2], argv[3]);
	//captureFrame("../images/dirt2.mp4", "../images/frames_dirt/");
	return 0;
}