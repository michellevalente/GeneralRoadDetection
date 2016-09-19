#include "gaborKernel.h"
#include "roadDetector.h"
#include "roadDetectorPeyman.h"

int captureFrame(string src, string dst, int sample_interval = 10, 
	int width = 480, int height = 360, int fps = 30)
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
		sprintf(filename, "%simg%d.jpg", dst.c_str(), ++cnt);
		imwrite(string(filename), frame);		
	}

	return cnt;
	
}


void genVideo(string src, int numFrames, string output_name = "output.avi", 
	double fps = 10.0, int wFrame = 960, int hFrame = 290)
{
	//VideoWriter vw(output_name, CV_FOURCC('M','J','P','G'), fps, Size(wFrame, hFrame));
	// if (!vw.isOpened())
	// {
	// 	cout << "video writer not initialized" << endl;
	// 	return;
	// }
	RoadDetectorPeyman roadDetector;
	Point vp = Point(0 , 0);

	for (int i = 0; i < numFrames; i++)
	{		
		char filename[512];

		sprintf(filename, "%s%010d.png", src.c_str(), i);
		cout << endl << filename << endl;

		roadDetector.applyFilter(filename, wFrame , hFrame);
		roadDetector.calcOrientationConfiance();
		roadDetector.drawOrientation();
		vp = roadDetector.findVanishingPoint(vp);
		roadDetector.findRoad();
		
		imshow("", roadDetector.image);
		waitKey(10);
		//vw.write(roadDetector.image);
	}
}

// void testImgs(string img_dir, string res_dir)
// {
// 	Detector road_detector;
// 	vector<string> filelist;
// 	getFileList(img_dir, filelist);
// 	string img_dir_ = addSplash(img_dir);
// 	string res_dir_ = addSplash(res_dir);
// 	for (int i = 0; i < filelist.size(); i++)
// 	{
// 		cout << endl << filelist[i] << endl;

// 		Point van_point;
// 		vector<float> line_angles;
// 		road_detector.vanishPointDet("dust.jpg", van_point);
// 		// road_detector.edgeDet(van_point, line_angles);
// 		// road_detector.drawLines(van_point, line_angles);
// 		// oritation(road_detector.color_img, van_point);

// 		roadDetector.applyFilter();
// 		roadDetector.calcOrientationConfiance2();
// 		roadDetector.drawOrientation();
// 		Point vanishing_point = roadDetector.findVanishingPoint2();

// 		imshow("", road_detector.color_img);
// 		waitKey(10);
// 		imwrite("snow_res.jpg", road_detector.color_img);
// 		waitKey(0);
// 	}
// }

void testImage(string img_dir,int wFrame = 480, int hFrame = 360)
{
	RoadDetectorPeyman roadDetector;
	Point vp = Point(0,0);
	roadDetector.applyFilter(img_dir,wFrame, hFrame);
	roadDetector.calcOrientationConfiance();
	roadDetector.drawOrientation();
	roadDetector.findVanishingPoint(vp);
	//roadDetector.findRoad();

	imshow("Final image", roadDetector.image);
	waitKey(0);
}


int main()
{
	//captureFrame("../data/test/");
	//testImage("../images/image_02/data/0000000000.png");
	//testImage("../images/dust3.jpg");
	genVideo("../images/image_02/data/", 188, "test_results/test.avi");
	//captureFrame("../images/rimg.mov", "../images/frames/");
	return 0;
}

// int main(int argc,char *argv[])
// {
// 	if(argc < 2)
// 	{
// 		cout << "Enter file name" << endl;
// 		return 0;
// 	}

// 	string fileName(argv[1]);


// 	RoadDetectorPeyman roadDetector("../images/" + fileName);
// 	roadDetector.applyFilter();
// 	roadDetector.calcOrientationConfiance();
// 	roadDetector.drawOrientation();
// 	roadDetector.findVanishingPoint();
// 	//roadDetector.findRoad();

// 	imshow("Final image", roadDetector.image);
//     waitKey(0);

// 	return 0;
// }