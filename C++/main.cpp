#include "gaborKernel.h"
#include "roadDetector.h"

int main(int argc,char *argv[])
{
	if(argc < 2)
	{
		cout << "Enter file name" << endl;
		return 0;
	}

	string fileName(argv[1]);
	cout << fileName << endl;
	RoadDetector roadDetector("../images/" + fileName, 0.7, 18, 36, 6, 480, 360);
	roadDetector.applyFilter();
	roadDetector.calcOrientationConfiance();
	roadDetector.drawOrientation();
	//roadDetector.drawConfidence();
	roadDetector.findVanishingPoint();
	
	//roadDetector.findVanishingPoint();
	return 0;
}