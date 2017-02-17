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
	double fps, int wFrame , int hFrame, char * showOrientation , char * showRoad,  char * regionGrowing)
{
	VideoWriter vw("out.avi", CV_FOURCC('M','J','P','G'), fps, Size(wFrame, hFrame));
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
        	// if(imgName.find("uu_") == std::string::npos)
        	// 	continue;
        	
           if(imgName.compare(".")!= 0 && imgName.compare("..")!= 0 && imgName.compare(".DS_Store")!= 0 && imgName.find("right") == std::string::npos)
           {
             string aux;
             aux.append(src);
             aux.append(imgName);
             cout << aux << endl;

            clock_t start;
			start = std::clock();

			roadDetector.applyFilter(aux, wFrame , hFrame);
			roadDetector.calcOrientationConfiance();
			if(!strcmp(showOrientation, "1"))
				roadDetector.drawOrientation();

			// Reset of temporal method evert 10 frames
			if(count_frames%10 == 0)
				vp = Point(0,0);

			vp = roadDetector.findVanishingPoint(Point(0,0));


			//roadDetector.direction(vp);

			//if(strcmp(showRoad, "0"))
			vp = roadDetector.findRoad();

			vps.push_back(vp);
			
			//roadDetector.detectSky();
			if(strcmp(regionGrowing, "0"))
				roadDetector.roadDetection(atoi(regionGrowing));
			
			double duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;
			//cout << "--- Total time: " << duration << endl;

			imshow(imgName, roadDetector.outimage);
			imwrite( "../Results_Images/out.png", roadDetector.outimage );
			waitKey(0);
			vw.write(roadDetector.outimage);
			count_frames++;
			//cout << "NUM FRAME: " << count_frames << endl;
			if(count_frames == 200)
				break;
           }
	    }
        closedir (dir);
    } else {
        cout<<"No image"<<endl;
    }
}
void testImage(string img_dir,int wFrame, int hFrame , char * showOrientation , char * showRoad, char * regionGrowing )
{
	RoadDetectorPeyman roadDetector;
	Point vp = Point(0,0);
	roadDetector.applyFilter(img_dir,wFrame, hFrame);
	roadDetector.calcOrientationConfiance();
	if(!strcmp(showOrientation, "1"))
			roadDetector.drawOrientation();

	vp = roadDetector.findVanishingPoint(vp);

	if(strcmp(showRoad, "0"))
		vp = roadDetector.findRoad();

	//roadDetector.detectSky();
	if(strcmp(regionGrowing, "0"))
		roadDetector.roadDetection(atoi(regionGrowing));

	imshow("Final image", roadDetector.outimage);
	imwrite( "../Results_Images/google/result.png", roadDetector.outimage );
	waitKey(0);
}

void testVideo(string src, double fps , int wFrame , int hFrame, char * showOrientation, char * showRoad, char * regionGrowing)
{
	cout << src << endl;
	VideoCapture cap(src);
	double total_gabor = 0, total_orientation = 0, total_vp = 0, total_edge = 0, total_region = 0, total_total = 0;
    if(!cap.isOpened())  // check if we succeeded
        return ;

    VideoWriter vw("out.avi", CV_FOURCC('M','J','P','G'), fps, Size(wFrame, hFrame));
	if (!vw.isOpened())
	{
		cout << "video writer not initialized" << endl;
		return;
	}

    Mat edges;
    RoadDetectorPeyman roadDetector;
	Point vp = Point(0,0);
	int count_frames = 0;

    for(;;)
    {
        Mat frame;
        cap.read(frame);
        if(!frame.empty())
        {
            //clock_t start, start2;
			//start = std::clock();
			//start2 = std::clock();
			roadDetector.applyFilter(frame,wFrame, hFrame);
			//double duration = ( std::clock() - start2 ) / (double) CLOCKS_PER_SEC;
			//total_gabor += duration;
			//cout << "--- GABOR time: " << duration << endl;
			//start2 = std::clock();
			roadDetector.calcOrientationConfiance();
			//duration = ( std::clock() - start2 ) / (double) CLOCKS_PER_SEC;
			//total_orientation += duration;
			//cout << "--- ORIENTATION time: " << duration << endl;
			if(!strcmp(showOrientation, "1"))
					roadDetector.drawOrientation();
			//start2 = std::clock();
			vp = roadDetector.findVanishingPoint(vp);
			//duration = ( std::clock() - start2 ) / (double) CLOCKS_PER_SEC;
			//total_vp += duration;
			//cout << "--- VP time: " << duration << endl;

			//start2 = std::clock();
			if(strcmp(showRoad, "0"))
				vp = roadDetector.findRoad();

			//duration = ( std::clock() - start2 ) / (double) CLOCKS_PER_SEC;
			//total_edge += duration; 
			//cout << "--- EDGE time: " << duration << endl;

			//start2 = std::clock();
			if(strcmp(regionGrowing, "0"))
				roadDetector.roadDetection(atoi(regionGrowing));

			//duration = ( std::clock() - start2 ) / (double) CLOCKS_PER_SEC;
			//total_region += duration;
			//cout << "--- REGION time: " << duration << endl;

			//duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;
			//total_total += duration;
			//cout << "--- Total time: " << duration << endl;
			imshow("", roadDetector.outimage);
			//if(count_frames % 10 == 0)
			//	imwrite( "../Results_Images/mojave/5/" + std::to_string(count_frames) + ".png", roadDetector.outimage );
			
			imwrite( "../Results_Images/out.png", roadDetector.outimage );
			waitKey(0);
			vw.write(roadDetector.outimage);
			count_frames++;

			// if(count_frames == 50){
			// 	cout << "GABOR: " << double(total_gabor / count_frames) << endl;
			// 	cout << "ORIENTATION: " << double(total_orientation / count_frames) << endl;
			// 	cout << "VP: " << double(total_vp / count_frames) << endl;
			// 	cout << "EDGE: " << double(total_edge / count_frames) << endl;
			// 	cout << "REGION: " << double(total_region / count_frames) << endl;
			// 	cout << "TOTAL: " << double(total_total / count_frames) << endl;
			// 	break;
			// }

			if(count_frames == 400)
				break;
		}
    }
    // the camera will be deinitialized automatically in VideoCapture destructor
    return ;
}


int main(int argc,char *argv[])
{
	if(argc < 5){
		cout << "Missing arguments!" << endl;
		return 1;
	}

	if(!strcmp(argv[1],"road"))
		genVideo("../images/road/data/", 40, "test_results/road.avi", 10, 960, 290, argv[2], argv[3], argv[4]);
	if(!strcmp(argv[1],"kitti"))
		genVideo("../kitti/data_road/testing/image_2/", 300, "test_results/road.avi", 10, 960, 290, argv[2], argv[3], argv[4]);
	if(!strcmp(argv[1],"road2"))
		genVideo("../images/2011_09_26_2/2011_09_26_drive_0001_sync/image_02/data/", 107, "test_results/road2.avi", 10, 840, 254, argv[2], argv[3], argv[4]);
	if(!strcmp(argv[1],"road3"))
		genVideo("../images/2011_09_26_3/2011_09_26_drive_0028_sync/image_02/data/", 40, "test_results/road3.avi", 5, 840, 254, argv[2], argv[3], argv[4]);
	if(!strcmp(argv[1],"road4"))
		genVideo("../images/2011_09_28/2011_09_28_drive_0047_sync/image_01/data/", 60, "test_results/road3.avi", 5, 840, 254, argv[2], argv[3], argv[4]);
	else if(!strcmp(argv[1],"dirt"))
		genVideo("../images/frames_dirt/", 110, "test_results/dirt.avi", 10, 360, 240, argv[2], argv[3], argv[4]);
	else if(!strcmp(argv[1],"dirt2"))
		genVideo("../images/frames_dirt2/", 50, "test_results/dirt2.avi", 5, 320, 240, argv[2], argv[3], argv[4]);
	else if(!strcmp(argv[1],"amelie"))
		genVideo("../images/cam/", 800, "test_results/amelie.avi", 10, 320, 240, argv[2], argv[3], argv[4]);
	else if(!strcmp(argv[1],"cordova"))
		genVideo("../images/caltech-lanes/cordova1/", 100, "test_results/cordova1.avi", 10, 640,480, argv[2], argv[3], argv[4]);
	else if(!strcmp(argv[1],"cordova2"))
		genVideo("../images/caltech-lanes/cordova2/", 405, "test_results/cordova2.avi", 10, 640, 480, argv[2], argv[3], argv[4]);
	else if(!strcmp(argv[1],"washington"))
		genVideo("../images/caltech-lanes/washington1/", 405, "test_results/washington1.avi", 10, 640, 480, argv[2], argv[3], argv[4]);
	else if(!strcmp(argv[1],"washington2"))
		genVideo("../images/caltech-lanes/washington2/", 405, "test_results/washington2.avi", 10, 640, 480, argv[2], argv[3], argv[4]);
	else if(!strcmp(argv[1],"test"))
		genVideo("../images/OpenCV_GoCART/bin/groundtruth_images/", 1000, "test_results/test.avi", 10, 640, 480, argv[2], argv[3], argv[4]);
	else if(!strcmp(argv[1], "snow"))
		testImage("../images/snow.jpg", 480,360, argv[2], argv[3], argv[4]);
	else if(!strcmp(argv[1], "my_road"))
		testImage("../images/my_road.jpg", 480,360, argv[2], argv[3], argv[4]);
	else if(!strcmp(argv[1], "cap"))
		captureFrame("../images/dirt2.mp4", "../images/frames_dirt2/");
	else if(!strcmp(argv[1], "moj"))
		testVideo("../images/OpenCV_GoCART_2/bin/teste4.mp4", 10, 360, 240, argv[2], argv[3], argv[4]);
	else if(!strcmp(argv[1], "mojave"))
		testVideo("../images/mojave.mp4", 10, 360, 240, argv[2], argv[3], argv[4]);
	else if(!strcmp(argv[1], "mojave2"))
		testVideo("../images/mojave2.mp4", 10, 360, 240, argv[2], argv[3], argv[4]);
	else if(!strcmp(argv[1], "mojave3"))
		testVideo("../images/mojave/stereo-sunday001.mp4", 10, 360, 240, argv[2], argv[3], argv[4]);
	else if(!strcmp(argv[1], "mojave4"))
		testVideo("../images/mojave/stereo-sunday002.mp4", 10, 360, 240, argv[2], argv[3], argv[4]);
	else if(!strcmp(argv[1], "desert"))
		testVideo("../images/desert.mp4", 10, 360, 240, argv[2], argv[3], argv[4]);
	else if(!strcmp(argv[1], "image")){
		string nameImage;
		cout << "Enter the name of the image: ";
		cin >> nameImage;
		testImage("../images/" + nameImage, 240,180, argv[2], argv[3], argv[4]);
	}
	return 0;
}