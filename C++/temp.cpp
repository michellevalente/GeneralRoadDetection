#include "roadDetectorPeyman.h"

/* Constructor */ 
RoadDetectorPeyman::RoadDetectorPeyman()
{
    perc_look_old = 0.07;
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

/* Distance between two points */
float distanceP(Point a, Point b)
{
    float difx = (a.x - b.x);
    float dify = (a.y - b.y);
    return sqrt(difx*difx  + dify*dify);
}

/* Check if point c is between point a and b */
bool isBetween(Point a, Point c, Point b)
{
    float dif = distanceP(a,c) + distanceP(c,b) - distanceP(a,b);
    return (dif <= 0.001 && dif >= -0.001);
}

/* Get in which point a line intersects another one */
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

/* Apply Gabor Filter to the image */
void RoadDetectorPeyman::applyFilter(std::string dfilename, int dw, int dh)
{

    filename = dfilename;
    w = dw;
    h = dh;
    float middle_x = w / 2.0;
    float middle_y = h / 2.0;

    /* Create region of interest in the image */
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

    /* Get image, resize and change to gray scale */
    image = imread(filename);
    resize(image, image, Size(w, h));
    outimage = image.clone();
    cvtColor(image, imageGray, CV_BGR2GRAY);

    // Start counting the time
    clock_t start;
    start = std::clock();

    int numScales = 1;

    cout << "TESTE" << endl;
    cout << responseOrientation << endl;

    responseOrientation.resize(numOrientations);

    cout << "TESTE" << endl;


    /* Get the gabor energy response for each orientation */
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

void RoadDetectorPeyman::applyFilter(Mat file, int dw, int dh)
{
    w = dw;
    h = dh;
    float middle_x = w / 2.0;
    float middle_y = h / 2.0;

    /* Create region of interest in the image */
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

    /* Get image, resize and change to gray scale */
    image = file;
    resize(image, image, Size(w, h));
    cvtColor(image, imageGray, CV_BGR2GRAY);

    outimage = image.clone();

    // Start counting the time
    clock_t start;
    //cout << "Starting apply gabor filter" << endl;
    start = std::clock();
    int numScales = 1;

    responseOrientation.resize(numOrientations);

    /* Get the gabor energy response for each orientation */
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

/* Calculates the dominant orientation for each pixel */
void RoadDetectorPeyman::calcOrientationConfiance()
{
    theta = Mat::zeros(h, w, CV_32F);
    conf = Mat::zeros(h, w, CV_32F);
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

            vector<float> sortedResponse(response);
            std::sort(sortedResponse.begin(), sortedResponse.end(), greater<float>());
            conf.at<float>(i,j)  = 1 - (sortedResponse[1] / sortedResponse[0]);

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

    // Normalize
    double minConf ;
    double maxConf ;
    minMaxLoc(conf, &minConf, &maxConf);

    for(int k =0 ; k < h; k++)
        for(int z = 0; z < w; z++){
            conf.at<float>(k,z) = float(conf.at<float>(k,z) - minConf) / float(maxConf - minConf);
        }
}

/* Returns in which quadrant belongs the angle */
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


/* Change angle to be between 0 and 360 */
double constrainAngle(float x){
    float angleDeg = x * 180.0 / CV_PI;
    angleDeg = fmod(angleDeg,360);
    if (angleDeg < 0)
        angleDeg += 360;
    return angleDeg * CV_PI / 180.0;
}

/* Vanishing point voter */
void RoadDetectorPeyman::voter(const Point p, Mat& votes, Mat& dist_table, Point old_vp)
{
    float angle = theta.at<float>(p);
    angle = constrainAngle(angle);
    int quad = whichQuadrant(angle);

    /* Calculates the right direction of the ray */
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

    /* Checks which plane the ray intersects */
    Point plane;
    get_line_intersection(p, Point(x1,y1), Point(0,h), Point(w,h), plane);
    get_line_intersection(p, Point(x1,y1), Point(w,h), Point(w,0), plane);
    get_line_intersection(p, Point(x1,y1), Point(w,0), Point(0,0), plane);
    get_line_intersection(p, Point(x1,y1), Point(0,0), Point(0,h), plane);

    float sinTheta = abs(sinAngle);
    float distance;
    float distanceFunction; 
    double maxDistance = distanceP(p, plane);
    double variance = 0.25;

    float x = p.x + xStep;
    float y = p.y - yStep;

    while(y > margin_h && y < h - margin_h && x > margin_w && x < w - margin_w)
    {
        /* If not the first time calculating the vanishing point, 
           only looks around the old one */
        if(old_vp.x != 0 && old_vp.y != 0)
        {

            if(y > old_vp.y - int(perc_look_old *h) && y < old_vp.y + int(perc_look_old *h) && x > old_vp.x - int(perc_look_old *w) && x < old_vp.x + int(perc_look_old *w)){
                votes.at<float>(y,x) += (sinTheta);
                distance = distanceP(p, Point(y,x)) / maxDistance;
                distanceFunction = exp(-(distance * distance) / (2.0 * variance ) );
                votes.at<float>(y,x) += distanceFunction * sinTheta ;
            }
            else
                votes.at<float>(y,x) = 0.0;
        }
        /* First time calculating the vanishing point */
        else
        {
            votes.at<float>(y,x) += (sinTheta);
            distance = distanceP(p, Point(y,x)) / maxDistance;
            distanceFunction = exp(-(distance * distance) / (2.0 * variance ) );
            votes.at<float>(y,x) += distanceFunction * sinTheta ;
        }
    	x += xStep;
    	y -= yStep;
    }
}

/* Find vanishing point based on vanishing point from previous frame */
Point RoadDetectorPeyman::findVanishingPoint(Point old_vp)
{
    votes = Mat::zeros(h, w, CV_32F);
    int maxH = 0.9 * h;
    Mat dist_table;

    // Start counting time
    clock_t start;
    double duration;
    start = std::clock();

    if(old_vp.x != 0 && old_vp.y != 0)
    {
        //rectangle(image, Point(old_vp.x - (perc_look_old * w), 
        //    old_vp.y - (perc_look_old *h)), Point(old_vp.x + (perc_look_old *w), old_vp.y + (perc_look_old * h)), Scalar(255, 0, 255));
    }

    for (int i = margin_h ; i < h - margin_h ; i++)
    {
        for (int j = margin_w  ; j < w - margin_w ; j++)
        {
            if(conf.at<float>(i,j) > 0.5)
                voter(Point(j,i), votes, dist_table, old_vp);
            //voter(Point(500,200), votes, dist_table);
        }
    }

    duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;
    cout << "voter time " << duration << endl;

    double min_val, max_val;
    Point min_p, max_p;
    minMaxLoc(votes, &min_val, &max_val, &min_p, &max_p);

//     Scalar mean, stddev;
//     meanStdDev ( votes, mean, stddev );
//     std::cout << "mean" <<mean << std::endl;
//     std::cout << "stdev" <<stddev << std::endl;
//     //cout << "max value voter: "<< max_val << endl;

// for (int i = margin_h ; i < h - margin_h ; i++)
//     {
//         for (int j = margin_w  ; j < w - margin_w ; j++)
//         {
//             //cout << votes.at<float>(i,j) << endl;
//              if( votes.at<float>(i,j) > mean[0] )
//                  circle(image, Point(j,i), 6, Scalar(255, 0, 255), -1);
//         }

//     }

    circle(outimage, max_p, 6, Scalar(255, 0, 255), -1);
    vp = max_p;

    return max_p;
}

void RoadDetectorPeyman::drawConfidence()
{

    for(int i = 0 ; i < h ; i++)
    {
        for(int j = 0 ; j < w; j++)
        {
            if(conf.at<float>(i,j) > 0.5){
                outimage.at<cv::Vec3b>(i,j)[0] = 255;
                outimage.at<cv::Vec3b>(i,j)[1] = 0;
                outimage.at<cv::Vec3b>(i,j)[2] = 255;
            }
        }
    }
}

// Draw arrow in the direction of vanishing point
void drawDirectionArrow(Mat &img, const Point &p_start, double theta, 
    Scalar color = Scalar(40, 240, 160), double len = 200.0, double alpha = CV_PI / 6)
{
    Point p_end;
    p_end.x = p_start.x + int(len * cos(theta));
    p_end.y = p_start.y - int(len * sin(theta));
    line(img, p_start, p_end, color, 3);
    double len1 = len * 0.1;
    Point p_arrow;
    p_arrow.x = p_end.x - int(len1 * cos(theta - alpha));
    p_arrow.y = p_end.y + int(len1 * sin(theta - alpha));
    line(img, p_end, p_arrow, color, 3);
    p_arrow.x = p_end.x - int(len1 * cos(theta + alpha));
    p_arrow.y = p_end.y + int(len1 * sin(theta + alpha));
    line(img, p_end, p_arrow, color, 3);
}
void RoadDetectorPeyman::direction(Point van_point)
{
    double theta = atan2(h - van_point.y, van_point.x - w / 2);
    drawDirectionArrow(outimage, Point(w / 2, h), theta);
}

// Draw dominant orientation for pixels in the image
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
            line(outimage, Point(j,i), Point(x,y), Scalar(255,100,40), 2);
            circle(outimage, Point(j,i), 1, Scalar(255,255,0), -1);
        }
    }
}

// Get the OCR for each edge
float RoadDetectorPeyman::voteRoad(float angle, float thr, Point p)
{
    float angleRad = angle * CV_PI / 180.0;
    int step = 3;
    float xStep = (step * cos(angleRad));
    float yStep = (step * sin(angleRad));

    float x = p.x + xStep;
    float y = p.y - yStep;

    int totalPoints = 0;
    float total = 0.0;
    float dif = 0.0;
    while(y > 0 && y < h && x > 0 && x < w)
    {
        if(abs(theta.at<float>(y,x) -  (angleRad)) < thr || abs(theta.at<float>(y,x) + CV_PI -  (angleRad)) < thr)
            total += 1;
        x += xStep;
        y -= yStep;
        totalPoints += 1;
    }
    if(totalPoints == 0)
        return 0;
    else
        return float(total / totalPoints);
}

// Compute all the OCRs
float RoadDetectorPeyman::computeOCR(vector<float>& voteEdges, Point point, int initialAngle , int & sumTopOCR)
{
    int i = 0;
    int initialLeft = 200;
    int finalRight = -10;
    vector<float> tempVotes(voteEdges);

    if(initialAngle > 180 && initialAngle < 275)
        initialLeft = initialAngle + 20;
    else if(initialAngle >= 275 && initialAngle <= 360)
        finalRight = initialAngle - 360 - 20; 

    for(int angleLeft = initialLeft ; angleLeft < 275; angleLeft += 5)
    {
        voteEdges[i] = voteRoad(angleLeft, 0.15, point);
        tempVotes[i] = voteEdges[i];
        i++;
    }
    for(int angleRight = -90; angleRight < finalRight; angleRight +=5)
    {
        voteEdges[i] = voteRoad(angleRight, 0.15, point);
        tempVotes[i] = voteEdges[i];
        i++;
    }
    int maxIndex = max_element(voteEdges.begin(), voteEdges.end()) - voteEdges.begin();
    float bestAngle = (initialLeft + (5 * maxIndex)) * CV_PI / 180.0;

    std::sort(tempVotes.begin(), tempVotes.end(), greater<float>());

    sumTopOCR = 0;
    for(int j = 0; j < 8; j++){
        sumTopOCR += tempVotes[j];
    }

    return bestAngle;
}

// Algorithm to detect road 
void RoadDetectorPeyman::findRoad()
{
    clock_t start;
    double duration;

    //cout << "Starting Find Road" << endl;
    start = std::clock();

    float angle;
    int diag = h + w;
    vector<float> voteEdges(50,0);
    int sumOCR;
    float bestAngle = computeOCR(voteEdges, vp, 200, sumOCR);
    float bestAngleDeg = bestAngle * 180.0 / CV_PI;
    int quadBest = whichQuadrant(bestAngle);

    float secondBestAngle = bestAngle;

    int i = 0;
    float maximum = 0;
    for(int angle = 200 ; angle < 350; angle += 5)
    {
        int quadNew = whichQuadrant(angle * CV_PI / 180.0);
        if(abs(bestAngleDeg - angle) > 30 &&  abs(bestAngleDeg - angle) < 150 && voteEdges[i] > maximum && quadBest != quadNew)
        {
            maximum = voteEdges[i];
            secondBestAngle = angle * CV_PI / 180.0;
        }

        i++;
    }

    int x1 = vp.x + int( diag * cos(bestAngle));
    int y1 = vp.y - int( diag * sin(bestAngle));
    line(outimage, Point(x1, y1), vp, Scalar(255,100,40), 2);

    int x2 = vp.x + int( diag * cos(secondBestAngle));
    int y2 = vp.y - int( diag * sin(secondBestAngle));
    line(outimage, Point(x2, y2), vp, Scalar(255,100,40), 2);
    int npt[] = { 3 };

    Point points[1][3];
    points[0][0] = vp;
    points[0][1] = Point(x1, y1);
    points[0][2] = Point(x2,y2);

    const Point* ppt[1] = { points[0] };

    float alpha = 0.3;

    Mat overlay;
    outimage.copyTo(overlay);
    fillPoly(overlay, ppt, npt, 1, Scalar( 255, 255, 0, 100 ), 8);
    addWeighted(overlay, alpha, outimage, 1 - alpha, 0, outimage);

    duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;
    cout << "finding road time " << duration << endl;
}

// Algorithm to detect road
void RoadDetectorPeyman::findRoad2()
{
    clock_t start;
    double duration;

    //cout << "Starting Find Road" << endl;
    start = std::clock();

    float angle;
    int diag = h + w;
    vector<float> voteEdges(50,0);
    vector<int> sumOCR(diag,0);
    vector<Point> points(diag);
    vector<float> angles(diag);
    int sum = 0;
    float bestAngle = computeOCR(voteEdges, vp, 200, sum);
    float bestAngleDeg = bestAngle * 180.0 / CV_PI;
    int step = 3;

    float xStep = (step * cos(bestAngle));
    float yStep = (step * sin(bestAngle));

    float x = vp.x;
    float y = vp.y;


    int i = 0;
    while(y > margin_h && y < h - h/3 && x > margin_w && x < w - margin_w)
    {
        vector<float> voteNewEdges(50,0);
        float bestAngleEdge = computeOCR(voteNewEdges, Point(x,y),bestAngleDeg, sumOCR[i]);
        float edgeDeg = bestAngleEdge * CV_PI / 180.0;
        x += xStep;
        y -= yStep;
        points[i] = Point(x,y);
        angles[i] = bestAngleEdge;
        i += 1;
    }

    x = vp.x ;
    y = vp.y ;

    while(y > margin_h - h/3 && y < h - h/3 && x > margin_w && x < w - margin_w)
    {
        vector<float> voteNewEdges(50,0);
        float bestAngleEdge = computeOCR(voteNewEdges, Point(x,y),bestAngleDeg, sumOCR[i]);
        float edgeDeg = bestAngleEdge * CV_PI / 180.0;
        x -= xStep;
        y += yStep;
        points[i] = Point(x,y);
        angles[i] = bestAngleEdge;
        i += 1;
    }
 
    int maxIndex = max_element(sumOCR.begin(), sumOCR.end()) - sumOCR.begin();
    circle(outimage, points[maxIndex], 6, Scalar(255, 0, 0), -1);
    int x1 = points[maxIndex].x + int( diag * cos(angles[maxIndex]));
    int y1 = points[maxIndex].y - int( diag * sin(angles[maxIndex]));
    line(outimage, Point(x1, y1), points[maxIndex], Scalar(255,100,40), 2);

    int x2 = points[maxIndex].x + int( diag * cos(bestAngle));
    int y2 = points[maxIndex].y - int( diag * sin(bestAngle));
    line(outimage, Point(x2, y2), points[maxIndex], Scalar(255,100,40), 2);

    int npt[] = { 3 };
    Point points_road[1][3];
    points_road[0][0] = vp;
    points_road[0][1] = Point(x1, y1);
    points_road[0][2] = Point(x2,y2);

    const Point* ppt[1] = { points_road[0] };

    float alpha = 0.3;

    Mat overlay;
    outimage.copyTo(overlay);
    fillPoly(overlay, ppt, npt, 1, Scalar( 255, 255, 0, 100 ), 8);
    addWeighted(overlay, alpha, outimage, 1 - alpha, 0, outimage);

    duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;
    cout << "finding road time " << duration << endl;
}


// Algorithm to detect sky in the image
void RoadDetectorPeyman::detectSky()
{
    Mat small_image;
    resize(image, small_image, Size(80, 60));
    cvtColor(small_image, small_image, CV_BGR2HSV);

    vector<Mat> hsv_planes;
    split( small_image, hsv_planes );

    float variance_height = h / 60;

    int i = 2;
    float mean_hue_top, mean_hue_bottom, diss_hue, diss_value;
    float mean_value_top, mean_value_bottom;
    float variance_hue_top, variance_hue_bottom, variance_value_top, variance_value_bottom;
    int count_top =0;
    int count_bottom =0;

    cout << (int) hsv_planes[0].at<uchar>(0,0) << endl;

    vector<float> distance_func(h,0);
    while(i < h)
    {
        float sum_hue_top = 0;
        float sum_hue_bottom = 0;
        float sum_value_top = 0;
        float sum_value_bottom = 0;

        // Calculate mean
        count_top = 0;
        count_bottom = 0;
        for (int y = 0 ; y < h; y++)
        {
            for (int x = 0  ; x < w; x++)
            {
                if(y < i)
                {
                    sum_hue_top += (int) hsv_planes[0].at<uchar>(y,x);
                    sum_value_top += (int) hsv_planes[2].at<uchar>(y,x);
                    count_top++;
                }
                else
                {
                    sum_hue_bottom += (int) hsv_planes[0].at<uchar>(y,x);
                    sum_value_bottom += (int) hsv_planes[2].at<uchar>(y,x);
                    count_bottom++;
                }
            }
        }

        mean_hue_top = float(sum_hue_top / count_top);
        mean_hue_bottom = float(sum_hue_bottom / count_bottom);
        mean_value_top = float(sum_value_top / count_top);
        mean_value_bottom = float(sum_value_bottom / count_bottom);

        float sum_variance_hue_top = 0 , sum_variance_value_top = 0, sum_variance_hue_bottom = 0, sum_variance_value_bottom =0;

        // Calculate variance
        for (int y = 0 ; y < h; y++)
        {
            for (int x = 0  ; x < w; x++)
            {
                if(y < i)
                {
                    sum_variance_hue_top += ((int)hsv_planes[0].at<uchar>(y,x) - mean_hue_top)
                                            * ((int)hsv_planes[0].at<uchar>(y,x) - mean_hue_top);
                    sum_variance_value_top += ((int)hsv_planes[2].at<uchar>(y,x) - mean_value_top)
                                              * ((int)hsv_planes[2].at<uchar>(y,x) - mean_value_top);
                }
                else
                {
                    sum_variance_hue_bottom += ((int)hsv_planes[0].at<uchar>(y,x) - mean_hue_bottom) 
                                                * ((int)hsv_planes[0].at<uchar>(y,x) - mean_hue_bottom);
                    sum_variance_value_bottom += ((int)hsv_planes[2].at<uchar>(y,x) - mean_value_bottom) 
                                                * ((int)hsv_planes[2].at<uchar>(y,x) - mean_value_bottom);
                }
            }
        }

        variance_hue_top = sum_variance_hue_top / count_top;
        variance_hue_bottom = sum_variance_hue_bottom / count_bottom;
        variance_value_top = sum_variance_value_top / count_top;
        variance_value_bottom = sum_variance_value_bottom / count_bottom;

        diss_hue =  ( (mean_hue_top - mean_hue_bottom) * (mean_hue_top - mean_hue_bottom) ) / 
                    (variance_hue_top + variance_hue_bottom); 

        diss_value =  ( (mean_value_top - mean_value_bottom) * (mean_value_top - mean_value_bottom) ) / 
                    (variance_value_top + variance_value_bottom); 

        distance_func[i] = (i / h) * diss_hue + diss_value;
        
        i++;
    }


    int maxIndex = max_element(distance_func.begin(), distance_func.end()) - distance_func.begin();
    cout << "Max index sky: " << maxIndex << endl;
    line(outimage, Point(0,int(maxIndex * variance_height)), Point(w, int(maxIndex * variance_height)), Scalar(255,100,40), 3);

}
bool RoadDetectorPeyman::pointIn(Point p) {
    return (0<p.x && p.x<w && 0<p.y && p.y< h);
}

static const Point shift[8]={Point(-1,0),Point(1,0),Point(0,-1),Point(0,1), Point(-2,0),Point(2,0),Point(0,-2),Point(0,2)};

float RoadDetectorPeyman::diffPixels(Point p, Point q, Mat img)
{
    Vec3b p1_h = channelHSV[0].at<Vec3b>(p);
    Vec3b p1_s = channelHSV[1].at<Vec3b>(p);
    Vec3b p2_h = channelHSV[0].at<Vec3b>(q);
    Vec3b p2_s = channelHSV[1].at<Vec3b>(q);

    cout << p1_h << endl;

    //return cv::norm(p1, p2, CV_L2); 
    //return sqrt((p1_h - p2_h) * (p1_h - p2_h)  + (p1_s - p2_s) * (p1_s - p2_s))
    return 0;
}   

void RoadDetectorPeyman::regionGrow(Point seed, double T, Mat img, Mat region, bool useOrientation)
{
    if (!pointIn(seed)) return;            
    queue<Point> active;        
    active.push(seed);    // add the seed

    while (!active.empty()) {
        Point p = active.front();
        active.pop();
        outimage.at<Vec3b>(p) = Vec3b(255,255,0);     // set region
        region.at<int>(p) = 1;
        int pointVp = 0;

        for(int i = 0; i < 8; i++)
        {
            
            Point q = p + shift[i];
            
            if(useOrientation){
                float thr = 0.1 * w;
                if(directionVp(q) < thr){
                    pointVp++;
                }
                for(int j = 0; j < 4;j++)
                {
                    Point w = q + shift[j];
                    if(directionVp(q) < thr){
                        pointVp++;
                        break;
                    }


                }
            }else
            {
                pointVp = 4.0; 
            }

            //cout << "Norm: " << diffPixels(p, q, img) << endl;
            if(pointIn(q) && diffPixels(p, q, img) < T && region.at<float>(q) == 0 &&  pointVp > 0)
            {
                active.push(q);
                region.at<int>(q) = 2;  
                outimage.at<Vec3b>(p) = Vec3b(255,255,0); 
            }
        }         
    }

}

float RoadDetectorPeyman::directionVp(Point p)
{
    float angle = theta.at<float>(p);
    int quad = whichQuadrant(angle);

    int step = 1;
    float xStep = (step * cos(angle));
    float yStep = (step * sin(angle));

    float x = p.x;
    float y = p.y;

    while(y > 0 && y < h && x > 0 && x < w)
    {
        //circle(outimage, Point(x,y), 6, Scalar(255, 0, 255), -1);
        if(int(y) == vp.y)
        {
            float distance = abs(vp.x - x);
            return distance;
        }
      
        if(yStep < 0){
            x -= xStep;
            y += yStep;
        }
        else{
            x += xStep;
            y -= yStep;
        }
    }

    return LONG_MAX;

}

void RoadDetectorPeyman::findLimits()
{
    Point left;
    Point right;
    for(int j = 0; j < w; j++)
    {
        for(int i = 0; i < h; i++)
        {
            if(outimage.at<Vec3b>(Point(j, i)) == Vec3b(255,255,0))
            {
                left = Point(j, i);
                break;
            }
        }
    }

    for(int j = w; j > 0 ; j--)
    {
        for(int i = 0; i < h; i++)
        {
            if(outimage.at<Vec3b>(Point(j, i)) == Vec3b(255,255,0))
            {
                right = Point(j, i);
                break;
            }
        }
    }

    line(outimage, vp, left, Scalar(255,100,40), 2);
    line(outimage, vp, right, Scalar(255,100,40), 2);
}

void RoadDetectorPeyman::roadDetection(float T)
{
    Point bottom(w/2, h);
    Point seed1 = (vp + bottom) * 0.5;
    Point seed2 = (vp + seed1) * 0.5;
    Point seed3 = (seed1 + bottom) * 0.5;
    //directionVp(Point(300,300));
    if(seed3.y > h - 0.2 * h)
        seed3.y = h - 0.2 * h;
    Mat imageDetection = image.clone();
    imageHSV = image.clone();
    channelHSV[0] = image.clone();
    channelHSV[1] = image.clone();
    channelHSV[2] = image.clone();
    cout << "TESTE" << endl;
    cvtColor(image, imageHSV, CV_BGR2HSV);
    split(imageHSV, channelHSV);
   // channel[2] = Mat(hsvImg.rows, hsvImg.cols, CV_8UC1, 200);//Set V
    //Merge channels
    //merge(channel, 3, hsvImg);
    //Mat rgbImg;
    //cvtColor(hsvImg, rgbImg, CV_HSV2BGR);
    //imshow("1. \"Remove Shadows\"", rgbImg);


    //GaussianBlur( imageGray, imageGray, Size( 3, 3 ), 0, 0 );
    GaussianBlur( image, imageDetection, Size( 17, 17 ), 0, 0 );

    //waitKey(0);

    //imageDetection = hsvImg;
   
    //medianBlur( imageGray, imageGray, 5 );
    //medianBlur( image, image, 5 );
    Mat region = Mat::zeros(h, w, CV_32F);
    for(int i = vp.y + h * 0.1; i < (h - h * 0.2); i ++)
        regionGrow(Point(seed1.x, i), T, imageDetection, region, 1);

    // for(int i = vp.y + h * 0.1; i < (h - h * 0.2); i ++)
    //    regionGrow(Point(seed1.x, i), T, outimage, region, 0);

   findLimits();
    // regionGrow(seed2, T, imageDetection);
    // regionGrow(seed3, T, imageDetection);

}
