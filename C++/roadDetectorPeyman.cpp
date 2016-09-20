#include "roadDetectorPeyman.h"


float distanceP(Point a, Point b)
{
    float difx = (a.x - b.x);
    float dify = (a.y - b.y);
    return sqrt(difx*difx  + dify*dify);
}

bool isBetween(Point a, Point c, Point b)
{
    float dif = distanceP(a,c) + distanceP(c,b) - distanceP(a,b);
    return (dif <= 0.001 && dif >= -0.001);
}

bool get_line_intersection(Point p0, Point p1, Point p2, Point p3, Point& i)
{
    float s1_x, s1_y, s2_x, s2_y;
    s1_x = p1.x - p0.x;     s1_y = p1.y - p0.y;
    s2_x = p3.x - p2.x;     s2_y = p3.y - p2.y;

    float s, t;
    s = (-s1_y * (p0.x - p2.x) + s1_x * (p0.y - p2.y)) / (-s2_x * s1_y + s1_x * s2_y);
    t = ( s2_x * (p0.y - p2.y) - s2_y * (p0.x - p2.x)) / (-s2_x * s1_y + s1_x * s2_y);

    if (s >= 0 && s <= 1 && t >= 0 && t <= 1)
    {
        // Collision detected
        i.x = p0.x + (t * s1_x);
        i.y = p0.y + (t * s1_y);
        return 1;
    }

    return 0; // No collision
}

RoadDetectorPeyman::RoadDetectorPeyman()
{
    numOrientations = 4;
    orientations.resize(numOrientations);
    orientations[0] = 0.0;
    orientations[1] = CV_PI / 4.0;
    orientations[2] = CV_PI / 2.0;
    orientations[3] = CV_PI * 3 / 4.0;

    kernels.resize(numOrientations);
    float Sigma = 2 * CV_PI;
    float F = sqrt(2.0);
    for(int i = 0; i < numOrientations; i++)
    {
        kernels[i].resize(1);
        kernels[i][0].init(Sigma, F, orientations[i], 0);
    }
}

void RoadDetectorPeyman::applyFilter(std::string dfilename, int dw, int dh)
{
    filename = dfilename;
    w = dw;
    h = dh;
    float middle_x = w / 2.0;
    float middle_y = h / 2.0;

    if(middle_y + 180 > h)
        margin_h = h * 0.1;
    else
        margin_h = h - (middle_y + 180);

    if(middle_x + 180 > w)
        margin_w = 0;
    else
        margin_w = w - (middle_x + 240);

    // margin_w = w * 0.2;
    // margin_h = h * 0.1;
    vp = Point(w/2 , h/2);

    image = imread(filename);
    resize(image, image, Size(w, h));
    cvtColor(image, imageGray, CV_BGR2GRAY);

    clock_t start;
    cout << "Starting apply gabor filter" << endl;
    start = std::clock();
    int numScales = 1;

    responseOrientation.resize(numOrientations);

    for(int i = 0; i < numOrientations ; i++)
    {
        vector< Mat > responseScale(numScales);
        for(int j = 0; j < numScales; j++)
            kernels[i][j].applyKernel(imageGray, responseScale[j]);

        Mat responseScaleSum = responseScale[0];
        for(int j = 0; j < numScales; j++)
            responseScaleSum = responseScaleSum + responseScale[j];

        responseOrientation[i] = responseScaleSum / numScales ; 
    }

    double duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;
    cout << "gabor filter time " << duration << endl;
}

void RoadDetectorPeyman::calcOrientationConfiance()
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

            dif1 = abs(response[0]) - abs(response[2]);
            if(dif1 > 0)
                orientation[0] = orientations[0];
            else
                orientation[0] = orientations[2];

            dif2 = abs(response[1]) - abs(response[3]);
            if(dif2 > 0)
                orientation[1] = orientations[1];
            else
                orientation[1] = orientations[3];

            x1 = cos(orientation[0]) * abs(dif1);
            y1 = sin(orientation[0]) * abs(dif1);
            x2 = cos(orientation[1]) * abs(dif2);
            y2 = sin(orientation[1]) * abs(dif2);

            theta.at<float>(i,j) = (atan((y1 + y2) / (x1 + x2)));
        }   
    }
}

int whichQuadrant(float angle)
{
    if(angle >= 0.0 && angle < CV_PI / 2.0)
        return 1;
    else if(angle >= CV_PI/2.0 && angle <= CV_PI)
        return 2;
    else if(angle > CV_PI && angle <= 3 * CV_PI / 2.0)
        return 3;
    else if(angle > 3* CV_PI/2.0 && angle <= 2*CV_PI)
        return 4;
    else
        return -1;
}

double constrainAngle(float x){
    float angleDeg = x * 180.0 / CV_PI;
    angleDeg = fmod(angleDeg,360);
    if (angleDeg < 0)
        angleDeg += 360;
    return angleDeg * CV_PI / 180.0;
}

void RoadDetectorPeyman::voter(const Point p, Mat& votes, Mat& dist_table, Point old_vp)
{
    float angle = theta.at<float>(p);
    angle = constrainAngle(angle);
    int quad = whichQuadrant(angle);

    if(p.x >= w/2)
    {
        if(quad == 1)
            angle += CV_PI;
        else if(quad == 4)
            angle -= CV_PI;
    }
    else
    {
        if(quad == 2)
            angle += CV_PI;
        else if(quad == 3)
            angle -= CV_PI;
    }

    float cosAngle = cos(angle);
    float sinAngle = sin(angle);
    int diag = h + w;

    int x1  = p.x + int( diag * cosAngle);
    int y1 = p.y - int( diag * sinAngle);

    int step = 3;
    float xStep = (step * cosAngle);
    float yStep = (step * sinAngle);

    Point plane;
    get_line_intersection(p, Point(x1,y1), Point(0,h), Point(w,h), plane);
    get_line_intersection(p, Point(x1,y1), Point(w,h), Point(w,0), plane);
    get_line_intersection(p, Point(x1,y1), Point(w,0), Point(0,0), plane);
    get_line_intersection(p, Point(x1,y1), Point(0,0), Point(0,h), plane);

    float sinTheta = abs(sinAngle);
    float distance;
    float distanceFunction; 
    double maxDistance = distanceP(p, plane);
    double variance = 0.28;

    float x = p.x + xStep;
    float y = p.y - yStep;

    // cout << "Point : " <<  p << endl;
    // cout << "Plane: " << plane << endl;
    // cout << " Max Distance: " << maxDistance << endl;

    //circle(image, plane, 3, Scalar(255, 255, 0), -1);
    while(y > margin_h && y < h - margin_h && x > margin_w && x < w - margin_w)
    {
        if(old_vp.x != 0 && old_vp.y != 0)
        {
            if(y > old_vp.y - 5 && y < old_vp.y + 5 && x > old_vp.x - 5 && x < old_vp.x + 5)
            {
                votes.at<float>(y,x) += (sinTheta);
            }
        }
        else
        {
            votes.at<float>(y,x) += (sinTheta);
        }
        //circle(image, Point(x,y), 1, Scalar(255, 255, 255), -1);
        // cout << "Distance: " << distance << endl;
        // cout << "Max distance: " << maxDistance << endl;
        distance = distanceP(p, Point(y,x)) / maxDistance;
        distanceFunction = exp(-(distance * distance) / (2.0 * variance ) );
        //cout << "Func: " << distanceFunction << endl;
        votes.at<float>(y,x) += (sinTheta) * distanceFunction;
        //votes.at<float>(y,x) += (sinTheta);
        //cout << "Vote: " << votes.at<float>(y,x) << endl;
    	x += xStep;
    	y -= yStep;
    }
}

Point RoadDetectorPeyman::findVanishingPoint(Point old_vp)
{
    votes = Mat::zeros(h, w, CV_32F);
    int maxH = 0.9 * h;
    Mat dist_table;

    clock_t start;
    double duration;

    cout << "Starting voter" << endl;
    start = std::clock();

    for (int i = margin_h ; i < h - margin_h ; i++)
    {
        for (int j = margin_w  ; j < w - margin_w ; j++)
        {
            voter(Point(j,i), votes, dist_table, old_vp);
            //voter(Point(500,200), votes, dist_table);
        }
    }

    duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;
    cout << "voter time " << duration << endl;

    double min_val, max_val;
    Point min_p, max_p;
    minMaxLoc(votes, &min_val, &max_val, &min_p, &max_p);
    cout << max_val << endl;
    cout << max_p << endl;

// for (int i = margin_h ; i < h - margin_h ; i++)
//     {
//         for (int j = margin_w  ; j < w - margin_w ; j++)
//         {
//             //cout << votes.at<float>(i,j) << endl;
//              if( votes.at<float>(i,j) > 120 )
//                  circle(image, Point(j,i), 6, Scalar(255, 0, 255), -1);
//         }

//     }

    circle(image, max_p, 6, Scalar(255, 0, 255), -1);
    vp = max_p;

    return max_p;
}

void RoadDetectorPeyman::drawOrientation(int grid, int lenLine)
{
    int maxH = 0.9 * h;
    for(int i = margin_h; i < h -margin_h ; i += grid)
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
}

float RoadDetectorPeyman::voteRoad(float angle, float thr)
{
    float angleRad = angle * CV_PI / 180.0;
    int step = 3;
    float xStep = (step * cos(angleRad));
    float yStep = (step * sin(angleRad));

    float x = vp.x + xStep;
    float y = vp.y - yStep;

    //circle(image, p, 3, Scalar(255, 255, 0), -1);
    int totalPoints = 0;
    float total = 0.0;
    float dif = 0.0;
    while(y > 0 && y < h && x > 0 && x < w)
    {
        // if(angle <= 0)
        // {
        //     // cout << "DIF1: " << abs(theta.at<float>(y,x) - CV_PI / 2.0 -  (angleRad)) << endl;
        //     // cout << "DIF2: " << abs(theta.at<float>(y,x) + CV_PI / 2.0 -  (angleRad)) << endl;
        //     if(abs(theta.at<float>(y,x) - CV_PI / 2.0 -  (angleRad)) < thr || abs(theta.at<float>(y,x) + CV_PI / 2.0 -  (angleRad)) < thr)
        //         total += 1;
        // }
        // else
        // {
            // cout << "DIF1: " << abs(theta.at<float>(y,x) -  (angleRad)) << endl;
            // cout << "DIF2: " << abs(theta.at<float>(y,x) + CV_PI -  (angleRad)) << endl;
            if(abs(theta.at<float>(y,x) -  (angleRad)) < thr || abs(theta.at<float>(y,x) + CV_PI -  (angleRad)) < thr)
                total += 1;
        // }
        x += xStep;
        y -= yStep;
        totalPoints += 1;
    }
    return float(total / totalPoints);
}

void RoadDetectorPeyman::findRoad()
{
    float angle;
    int diag = h + w;
    vector<float> voteEdges(50,0);
    int i = 0;
    for(int angleLeft = 200 ; angleLeft < 275; angleLeft += 5)
    {
        voteEdges[i] = voteRoad(angleLeft, 0.15);
        i++;
    }
    for(int angleRight = -90; angleRight < -10; angleRight +=5)
    {
        voteEdges[i] = voteRoad(angleRight, 0.15);
        i++;
    }
    int maxIndex = max_element(voteEdges.begin(), voteEdges.end()) - voteEdges.begin();
    float bestAngle = (200 + (5 * maxIndex)) * CV_PI / 180.0;
    float bestAngleDeg = (200 + (5 * maxIndex));

    int quadBest = whichQuadrant(bestAngle);

    float secondBestAngle = bestAngle;

    i = 0;
    float maximum = 0;
    for(int angle = 200 ; angle < 350; angle += 5)
    {
        int quadNew = whichQuadrant(angle * CV_PI / 180.0);
        if(abs(bestAngleDeg - angle) > 20 &&  abs(bestAngleDeg - angle) < 100 && voteEdges[i] > maximum && quadBest != quadNew)
        {
            maximum = voteEdges[i];
            secondBestAngle = angle * CV_PI / 180.0;
        }

        i++;
    }

    int x1 = vp.x + int( diag * cos(bestAngle));
    int y1 = vp.y - int( diag * sin(bestAngle));
    line(image, Point(x1, y1), vp, Scalar(255,100,40), 2);

    int x2 = vp.x + int( diag * cos(secondBestAngle));
    int y2 = vp.y - int( diag * sin(secondBestAngle));
    line(image, Point(x2, y2), vp, Scalar(255,100,40), 2);
    int npt[] = { 3 };

    Point points[1][3];
    points[0][0] = vp;
    points[0][1] = Point(x1, y1);
    points[0][2] = Point(x2,y2);

    const Point* ppt[1] = { points[0] };

    float alpha = 0.3;

    Mat overlay;
    image.copyTo(overlay);
    fillPoly(overlay, ppt, npt, 1, Scalar( 255, 255, 0, 100 ), 8);
    addWeighted(overlay, alpha, image, 1 - alpha, 0, image);
}

