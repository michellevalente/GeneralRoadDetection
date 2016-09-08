#include "RoadDetector.h"

RoadDetector::RoadDetector(std::string dfilename, double dpercImgDetect, int dnumSeg, int dnumOrientations, int dnumScales, int dw, int dh)
{
	filename = dfilename;
	numSeg = dnumSeg;
	numOrientations = dnumOrientations;
	numScales = dnumScales;
	w = dw;
	h = dh;
    percImgDetect = dpercImgDetect;

	image = imread(filename);
	resize(image, image, Size(w, h));
    cvtColor(image, imageGray, CV_BGR2GRAY);

    double spareNum = (1 + numSeg - numOrientations) / 2.0;

    orientations.resize(numOrientations);
    if(numOrientations == 4)
    {
        orientations[0] = 0.0;
        orientations[1] = CV_PI / 4.0;
        orientations[2] = CV_PI / 2.0;
        orientations[3] = CV_PI * 3 / 4.0;
    }
    else
    {
        double delta = float(CV_PI / numSeg);
        for(int i = 0; i < numOrientations; i++){
            orientations[i] = float((i + spareNum) * delta);
        }
        initKernel();
    }
}

void RoadDetector::initKernel()
{
    kernels.resize(numOrientations);
    float Sigma = 2 * CV_PI;
    float F = sqrt(2.0);
    for(int i = 0; i < numOrientations; i++)
    {
        kernels[i].resize(numScales);
        for(int j = 0; j < numScales; j++)
            kernels[i][j].init(Sigma, F, orientations[i], j);
    }
}

void RoadDetector::applyFilter()
{
    clock_t start;
    cout << "Starting apply gabor filter" << endl;
    start = std::clock();

    responseOrientation.resize(numOrientations);

    #pragma omp parallel for
    for(int i = 0; i < numOrientations ; i++)
    {
        vector< Mat > responseScale(numScales);
        for(int j = 0; j < numScales; j++)
            kernels[i][j].applyKernel(imageGray, responseScale[j]);

        Mat responseScaleSum = responseScale[0];
        for(int j = 0; j < numScales; j++)
            responseScaleSum = responseScaleSum + responseScale[j];

        responseOrientation[i] = responseScaleSum / numScales ; 
        //cout << responseOrientation[i].at<float>(0,0) << endl;
    }

    double duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;
    cout << "gabor filter time " << duration << endl;
}

void RoadDetector::calcOrientationConfiance2()
{
    theta = Mat::zeros(h, w, CV_32F);
    Mat s = Mat::zeros(h, w, CV_32F);
    vector<float*> temp(numOrientations);
    double dif1, dif2, x1 , y1, x2, y2; 
    double orientation[2];
    for(int i =0 ; i < h; i++)
    {
        for(int orientation = 0 ; orientation < numOrientations; orientation++){
            temp[orientation] = responseOrientation[orientation].ptr<float>(i);
        }

        for(int j = 0; j < w ; j++)
        {
            vector<float> response(numOrientations);
            for(int k = 0; k < numOrientations; k++){
                response[k] = temp[k][j];
            }

            dif1 = response[0] - response[2];
            if(dif1 > 0)
                orientation[0] = orientations[0];
            else
                orientation[0] = orientations[2];

            dif2 = response[1] - response[3];
            if(dif2 > 0)
                orientation[1] = orientations[1];
            else
                orientation[1] = orientations[3];

            x1 = cos(orientation[0]) * dif1;
            y1 = sin(orientation[0]) * dif1;
            x2 = cos(orientation[1]) * dif2;
            y2 = sin(orientation[1]) * dif2;
            theta.at<float>(i,j) = atan((y1 + y2) / (x1 + x2));
        }
    }
}

void RoadDetector::calcOrientationConfiance()
{
    int temp1 = numOrientations / 9, temp2 = 15, temp3 = temp2 - temp1;
    theta = Mat::zeros(h, w, CV_32F);
    conf = Mat::zeros(h,w, CV_32F);
    vector<float*> temp(numOrientations);
    for(int i =0 ; i < h; i++)
    {
        for(int orientation = 0 ; orientation < numOrientations; orientation++){
            temp[orientation] = responseOrientation[orientation].ptr<float>(i);
        }

        for(int j = 0; j < w; j++)
        {
            vector<float> response(numOrientations);
            for(int k = 0; k < numOrientations; k++){
                response[k] = temp[k][j];
            }

            int maxIndex = max_element(response.begin(), response.end()) - response.begin();
            //cout << "Max Index: " << maxIndex << endl;
            theta.at<float>(i,j) = orientations[maxIndex]; 

            // Compute Confidence ( Miksik paper )

            int b = (numOrientations / 4) - 1;
            float sumResponse = 0;
            int count = 0;
            for(int k = maxIndex - b; k <= maxIndex + b; k++)
            {
                if(k < numOrientations){
                    sumResponse += response[k];
                    count += 1;
                }
            }

            float average = sumResponse / count;
            conf.at<float>(i,j) = 1 - (average / response[maxIndex]);

            for(int k = 0; k < maxIndex; k++)
                if(k < maxIndex - b || k > maxIndex +b)
                    if(average < response[k]){
                        conf.at<float>(i,j) = 0;
                        break;
                    }

            // Compute Confidence Kong paper
            // std::sort(response.begin(), response.end()); 
            // float sumResponse = 0;
            // for(int i = 4; i < 15; i++ ){
            //     sumResponse += response[i];
            // }
            // float average = sumResponse / 11;
            // cout << "Len : " << response.size() << endl;;
            // conf.at<float>(i,j) = 1 - (average / response[0]);
        }
    }

    double minConf ;
    double maxConf ;
    minMaxLoc(conf, &minConf, &maxConf);
    for(int k =0 ; k < h; k++)
        for(int z = 0; z < w; z++){
            conf.at<float>(k,z) = (conf.at<float>(k,z) - minConf) / (maxConf - minConf);
            //cout << conf.at<float>(k,z)  << endl;
        }
}

void RoadDetector::drawOrientation(int grid, int lenLine)
{
    int maxH = percImgDetect * h;
    for(int i = margin_h; i < maxH; i += grid)
    {
        for(int j = margin_w; j < w - margin_w; j += grid)
        {
            float angle = theta.at<float>(i,j);
            int x  = j + int( lenLine * cos(angle));
            int y = i - int( lenLine * sin(angle));
            line(image, Point(j,i), Point(x,y), Scalar(255,100,40), 2);
            circle(image, Point(j,i), 1, Scalar(255,255,0), -1);
        }
    }
    // imshow("orientation", image);
    // waitKey(0);
}

void RoadDetector::drawConfidence()
{
    for(int i = 0 ; i < h ; i++)
    {
        for(int j = 0 ; j < w; j++)
        {
            if(conf.at<float>(i,j) > 0.3){
                image.at<cv::Vec3b>(i,j)[0] = 255;
                image.at<cv::Vec3b>(i,j)[1] = 0;
                image.at<cv::Vec3b>(i,j)[2] = 255;
            }
        }
    }
    // imshow("confidence", image);
    // waitKey(0);
}

float RoadDetector::getScoreOrientation(Point p, float angle, int thr)
{
    int step = 3;
    float xStep = step * cos(angle);
    float yStep = step * sin(angle);
    int gap = h / (step * 16);
    int total = 0;
    int cor = 0;
    int difSum = 0;

    float x = p.x - gap * xStep;
    float y = p.y + gap * yStep;

    while(x > 0 && x < w && y > 0 && y < h)
    {
        total += 1;
        double dif = theta.at<float>(int(y), int(x)) - angle;
        difSum += dif;
        if(abs(dif) <= thr)
            cor += 1;
        x -= xStep;
        y += yStep;
    }

    if(total < int(0.3 * h / step))
        return -total;
    else
        return cor / sqrt(total);
}

float RoadDetector::getScore(Point point, float angleStep, float oriNum)
{
    int spare = (CV_PI / angleStep - oriNum + 1) / 2;
    vector<float> scores(oriNum);

    for(int i = 0; i < oriNum; i++)
    {
        double angle = (i + spare) * angleStep;
        scores[i] = getScoreOrientation(point, angle, angleStep);
    }

    std::sort(scores.begin(), scores.end(), greater<float>());
    float sum = .0f;
    for (int k = 0; k < 8; k++)
        sum += scores[k];
    return sum;      
}

float RoadDetector::getAngle(int x1,int y1,int x2,int y2,float gaborAngle)
{
  float rad = gaborAngle;
  int v1x = x2 - x1;
  int v1y = y2 - y1;
  float v2x = cos(rad);
  float v2y = sin(rad);
  float result  = atan2f(v2y,v2x) - atan2f(v1y,v1x);//9132 ms
  return result * 180.0/M_PI;
}

float RoadDetector::getScore2(int x,int y, float radius)
{
    float score = 0.0;

    //for each half disk
    for(int yy = 0;yy <= radius; ++yy)
    {
        int xbound = radius;
        for(int xx = -xbound; xx <= xbound; ++xx)
        {
            if(x + xx > 0 && x + xx < w && y + yy > 0 && y + yy < h)
            {
                float distance = sqrt(xx*xx + yy*yy);
                int gaborAngle = theta.at<float>(x+xx,y+yy); //optimize by using lookup table
                float pvoAngle = getAngle(x,y,x+xx,y+yy,gaborAngle);//make it faster by using sin/cos lookup table
                float kongsAngleThreshold = 5/(1+2*distance);
                float kongsScore = 1/(1+((pvoAngle*distance)*(pvoAngle*distance)));

                if(pvoAngle <= kongsAngleThreshold){
                    score+= kongsScore;
                }
            }
        }
    }
    return score;
}

Point RoadDetector::findVanishingPoint(int type, int segNum, int angleNum)
{
    double angleStep = CV_PI / segNum;
    votes = Mat::zeros(h, w, CV_32F);
    double radius = 0.35 * h;
    int maxH = percImgDetect * h;

    clock_t start;
    cout << "Starting voter to find vanishing point" << endl;
    start = std::clock();

    for (int i = 0; i < maxH; i++)
  {
    for (int j = 0 ; j < w; j++)
        {
            //if(conf.at<float>(i,j) > 0.3){
            //if((i % pixelGrid == 0 && j % pixelGrid == 0)){
                if(type ==0 )
                    votes.at<float>(i,j) = getScore(Point(j,i), angleStep, angleNum);
                else
                    votes.at<float>(i,j) = getScore2(j,i, radius);
            //}
        }
    }

    double duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;
    cout << "voter time " << duration << endl;

    double min_val, max_val;
    Point min_p, max_p;
    minMaxLoc(votes, &min_val, &max_val, &min_p, &max_p);
    circle(image, max_p, 6, Scalar(255, 255, 0), -1);

    imshow("vanishing point", image);
    waitKey(0);

    return max_p;
}

void RoadDetector::drawLines(const Point &van_point, const float lineAngle)
{
    int diag = h + w;
    // for (int i = 0; i < line_angles.size(); ++i)
    // {
        int temp_x = van_point.x - diag * cos(lineAngle);
        int temp_y = van_point.y + diag * sin(lineAngle);

        if(temp_y < van_point.y)
        {
            cout << "UPWARDS" << endl;
        }
        line(image, van_point, Point(temp_x, temp_y), Scalar(0, 0, 255), 3);
    // }
    circle(image, van_point, 6, Scalar(0, 255, 255), -1);
    imshow("Line", image);
    waitKey(0);
}


bool RoadDetector::liesOnRay(const Point &c, const Point &a, const Point &b)
{
    float crossproduct = (c.y - a.y) * (b.x - a.x) - (c.x - a.x) * (b.y - a.y);
    if( abs(crossproduct) > 0.01)
        return false;

    float dotproduct = (c.x - a.x) * (b.x - a.x) + (c.y - a.y)*(b.y - a.y);
    if (dotproduct < 0 )
        return false;

    float squaredlengthba = (b.x - a.x)*(b.x - a.x) + (b.y - a.y)*(b.y - a.y);
    if(dotproduct > squaredlengthba)
        return false;

    return true;
}

// bool RoadDetector::liesOnRay(const Point &p, const Point &begin, const Point &end)
// {
//     double temp1, temp2;
//     if((end.x - begin.x) == 0)
//         temp1 = 0;
//     else
//         temp1 = (p.x - begin.x) / (end.x - begin.x);

//     if((end.y - begin.y) == 0)
//         temp2 = 0;
//     else
//         temp2 = (p.y - begin.y) / (end.y - begin.y);

//     if(temp1 == temp2)
//         return true;
//     else
//         return false;
// }

float distancePoints(const Point p1, const Point p2)
{
    float x = p2.x - p1.x;
    float y = p2.y - p1.y;
    return sqrt(x*x + y*y);
}

void RoadDetector::getDistTable(Mat &dist_table)
{
    dist_table.create(h, w, CV_32F);
    for (int i = 0; i < h; i++)
    {
        float *ptr_dist = dist_table.ptr<float>(i);
        for (int j = 0; j < w; j++)
            ptr_dist[j] = sqrt(i * i + j * j);
    }
}

void RoadDetector::voter(const Point p, Mat& votes, Mat& dist_table)
{

    float angle = theta.at<float>(p);
    int diag = h + w;
    int x1 = p.x - diag * cos(angle);
    int y1 = p.y + diag * sin(angle); 

    //circle(image, Point(x1,y1), 10, Scalar(255, 255, 255), -1);
    //line(image, p,Point(x1,y1), Scalar(0, 0, 255), 3);
    if(y1 < p.y)
    {
        y1 = 0.0;
        x1 = (p.y + (p.x * tan(angle)) )/ tan(angle);
        //circle(image, Point(x1,y1), 6, Scalar(255, 255, 255), -1);
        //cout << "UPWARDS" << endl;
        float sinTheta = (sin(angle)) * 100;
        float distance;
        float distanceFunction; 
        double maxDistance = distancePoints(p, Point(x1,y1));
        double variance = 0.25;
        
        for(int i = margin_h; i <= p.y; i++)
        {
            //float *ptr_dist = dist_table.ptr<float>(i - p.x);
            for(int j = margin_w ; j < w - margin_w; j++)
            {
                if(liesOnRay(Point(j,i), p, Point(x1,y1))){
                    //circle(image, Point(j,i), 6, Scalar(255, 255, 0), -1);
                    //float dpv = ptr_dist[abs(j - p.y)];
                    //distance = distancePoints(p, Point(j,i)) / maxDistance;
                    //distance = dpv / maxDistance;
                    //distanceFunction = exp(-(distance * distance) / (2.0 * variance ) );
                    votes.at<float>(i,j) += (sinTheta) * distanceFunction;
                    //votes.at<float>(i,j) = votes.at<float>(i,j) +  1;
                    //votes.at<float>(i,j) += (sinTheta);
                    //votes.at<float>(i,j) += 1;
                }
            }
        }
    }
}

void RoadDetector::findVanishingPoint2()
{
    votes = Mat::zeros(h, w, CV_32F);
    int maxH = percImgDetect * h;
    Mat dist_table;

    clock_t start;
    double duration;
    // cout << "Starting creating distance table" << endl;
    // start = std::clock();
    // getDistTable(dist_table);

    // duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;
    // cout << "distance table time " << duration << endl;

    cout << "Starting voter" << endl;
    start = std::clock();

    for (int i = margin_h; i < maxH; i++)
    {
        for (int j = margin_w ; j < w - margin_w; j++)
        {
            voter(Point(j,i), votes, dist_table);
        }
    }

    duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;
    cout << "voter time " << duration << endl;

    double min_val, max_val;
    Point min_p, max_p;
    minMaxLoc(votes, &min_val, &max_val, &min_p, &max_p);
    cout << max_val << endl;
    cout << max_p << endl;
    circle(image, max_p, 6, Scalar(255, 0, 255), -1);

    // for (int i = 0; i < maxH /2; i++)
    // {
    //     for (int j = 0 ; j < w; j++)
    //     {

    //         if(votes.at<float>(i,j) > 0)
    //         {
    //             circle(image, Point(j,i), 6, Scalar(255, 255, 0), -1);
    //         }
    //     }
    // }

    //circle(image, max_p, 6, Scalar(255, 255, 0), -1);

    imshow("vanishing point", image);
    waitKey(0);
}

