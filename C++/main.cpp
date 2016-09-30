#include "gaborKernel.h"
#include "roadDetector.h"
#include "roadDetectorPeyman.h"
#include <dirent.h>
#include <string.h>

float roadDectection_T = 3.0;

int captureFrame(string src, string dst, int sample_interval = 10, 
	int width = 320, int height = 240, int fps = 10)
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


void genVideo(string src, int numFrames, string output_name, 
	double fps, int wFrame , int hFrame, char * showOrientation , char * showRoad)
{
	VideoWriter vw(output_name, CV_FOURCC('M','J','P','G'), fps, Size(wFrame, hFrame));
	if (!vw.isOpened())
	{
		cout << "video writer not initialized" << endl;
		return;
	}

	RoadDetectorPeyman roadDetector;
	Point vp = Point(0 , 0);

    DIR *dir;
    dir = opendir(src.c_str());
    string imgName;
    struct dirent *ent;
    vector<Point> vps(20);
    int count_frames = 0;
    if (dir != NULL) {
        while ((ent = readdir (dir)) != NULL) {

        	imgName= ent->d_name;
           if(imgName.compare(".")!= 0 && imgName.compare("..")!= 0 && imgName.compare(".DS_Store")!= 0 && imgName.find("right") == std::string::npos)
           {
             string aux;
             aux.append(src);
             aux.append(imgName);
             cout << aux << endl;

			roadDetector.applyFilter(aux, wFrame , hFrame);
			roadDetector.calcOrientationConfiance();
			if(!strcmp(showOrientation, "1"))
				roadDetector.drawOrientation();

			// Reset of temporal method evert 10 frames
			if(count_frames%10 == 0)
				vp = Point(0,0);

			vp = roadDetector.findVanishingPoint(vp);
			// for(auto const& point: vps) {
			// 	circle(roadDetector.image, point, 3, Scalar(255, 255, 255), -1);
			// }

			//roadDetector.direction(vp);

			vps.push_back(vp);

			if(!strcmp(showRoad,"1"))
				roadDetector.findRoad();

			if(!strcmp(showRoad,"2"))
				roadDetector.findRoad2();

			
			//roadDetector.detectSky();
			roadDetector.roadDetection(roadDectection_T);
			
			imshow("", roadDetector.image);
			waitKey(100);

			vw.write(roadDetector.image);
			count_frames++;
			if(count_frames == numFrames)
				break;
           }
	    }
        closedir (dir);
    } else {
        cout<<"No image"<<endl;
    }
}
void testImage(string img_dir,int wFrame, int hFrame , char * showOrientation , char * showRoad )
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

	if(!strcmp(showRoad,"2"))
		roadDetector.findRoad2();

	//roadDetector.detectSky();
	roadDetector.roadDetection(roadDectection_T);

	imshow("Final image", roadDetector.image);
	waitKey(0);
}

void testVideo(string src, double fps , int wFrame , int hFrame, char * showOrientation, char * showRoad)
{
	cout << src << endl;
	VideoCapture cap(src);
    if(!cap.isOpened())  // check if we succeeded
        return ;

    Mat edges;
    namedWindow("edges",1);
    RoadDetectorPeyman roadDetector;
	Point vp = Point(0,0);
    for(;;)
    {
        Mat frame;
        cap >> frame; // get a new frame from camera
        if(!frame.empty())
        {
			roadDetector.applyFilter(frame,wFrame, hFrame);
			roadDetector.calcOrientationConfiance();
			if(!strcmp(showOrientation, "1"))
					roadDetector.drawOrientation();

			vp = roadDetector.findVanishingPoint(vp);

			if(!strcmp(showRoad,"1"))
				roadDetector.findRoad();

			if(!strcmp(showRoad,"2"))
				roadDetector.findRoad2();

			roadDetector.roadDetection(roadDectection_T);
			imshow("", roadDetector.image);
			waitKey(10);
		}
    }
    // the camera will be deinitialized automatically in VideoCapture destructor
    return ;
}


int main(int argc,char *argv[])
{
	if(argc < 4){
		cout << "Missing arguments!" << endl;
		return 1;
	}

	if(argc == 5)
		roadDectection_T = std::stof(String(argv[4]));

	if(!strcmp(argv[1],"road"))
		genVideo("../images/road/data/", 120, "test_results/road.avi", 10, 960, 290, argv[2], argv[3]);
	if(!strcmp(argv[1],"road2"))
		genVideo("../images/2011_09_26_2/2011_09_26_drive_0001_sync/image_02/data/", 107, "test_results/road2.avi", 10, 840, 254, argv[2], argv[3]);
	if(!strcmp(argv[1],"road3"))
		genVideo("../images/2011_09_26_3/2011_09_26_drive_0028_sync/image_02/data/", 60, "test_results/road3.avi", 5, 840, 254, argv[2], argv[3]);
	else if(!strcmp(argv[1],"dirt"))
		genVideo("../images/frames_dirt/", 110, "test_results/dirt.avi", 10, 360, 240, argv[2], argv[3]);
	else if(!strcmp(argv[1],"dirt2"))
		genVideo("../images/frames_dirt2/", 50, "test_results/dirt2.avi", 5, 320, 240, argv[2], argv[3]);
	else if(!strcmp(argv[1],"amelie"))
		genVideo("../images/cam/", 800, "test_results/amelie.avi", 10, 320, 240, argv[2], argv[3]);
	else if(!strcmp(argv[1],"cordova"))
		genVideo("../images/caltech-lanes/cordova1/", 50, "test_results/cordova.avi", 10, 640, 480, argv[2], argv[3]);
	else if(!strcmp(argv[1],"malaga"))
		genVideo("../images/malaga-urban-dataset-extract-02/Images/", 240, "test_results/malaga.avi", 10, 640, 480, argv[2], argv[3]);
	else if(!strcmp(argv[1],"test"))
		genVideo("../images/OpenCV_GoCART/bin/groundtruth_images/", 1000, "test_results/test.avi", 10, 640, 480, argv[2], argv[3]);
	else if(!strcmp(argv[1], "snow"))
		testImage("../images/snow.jpg", 480,360, argv[2], argv[3]);
	else if(!strcmp(argv[1], "my_road"))
		testImage("../images/my_road.jpg", 480,360, argv[2], argv[3]);
	else if(!strcmp(argv[1], "cap"))
		captureFrame("../images/dirt2.mp4", "../images/frames_dirt2/");
	else if(!strcmp(argv[1], "mojave"))
		testVideo("../images/mojave.mp4", 10, 360, 240, argv[2], argv[3]);
	else if(!strcmp(argv[1], "mojave2"))
		testVideo("../images/mojave2.mp4", 10, 360, 240, argv[2], argv[3]);
	else if(!strcmp(argv[1], "desert"))
		testVideo("../images/desert.mp4", 10, 360, 240, argv[2], argv[3]);
	return 0;
}