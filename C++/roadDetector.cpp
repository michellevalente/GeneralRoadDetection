#include "RoadDetector.h"

double signOf(double x) { return (x < 0) ? -1 : (x > 0); }

bool insideImage(int x, int y, int h, int w) { return (x>= 0 && x < w && y >= 0 && y < h);}

RoadDetector::RoadDetector(std::string dfilename, double dpercImgDetect, int dnumSeg, int dnumOrientations, int dnumScales, int dw, int dh)
{
	filename = dfilename;
	numSeg = dnumSeg;
	numOrientations = dnumOrientations;
	numScales = dnumScales;
	w = dw;
	h = dh;
    percImgDetect = dpercImgDetect;
    margin_w = w * 0.2;
    margin_h = h * 0.2;

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
    }
    initKernel();
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
            theta.at<float>(i,j) = orientations[maxIndex]; 

            // Compute Confidence ( Miksik paper )

            // int b = (numOrientations / 4) - 1;
            // float sumResponse = 0;
            // int count = 0;
            // for(int k = maxIndex - b; k <= maxIndex + b; k++)
            // {
            //     sumResponse += response[k];
            //     count++;
            // }

            // float average = sumResponse / (float)count;
            // conf.at<float>(i,j) = 1 - (average / response[maxIndex]);

            // for(int k = 0; k < maxIndex; k++)
            //     if(k < maxIndex - b || k > maxIndex +b)
            //         if(average < response[k]){
            //             conf.at<float>(i,j) = 0;
            //             break;
            //         }

            // Compute Confidence Kong paper
            // std::sort(response.begin(), response.end(), std::greater<int>());
            // float sumResponse = 0;
            // cout << "Max: " << response[0] << endl;
            // cout << "Min: " << response[35] << endl;
            // int count = 0;
            // for(int i = 4; i < 15; i++ ){
            //     //cout << response[i] << endl;
            //     sumResponse += response[i];
            //     count++;
            // }

            // float average = sumResponse / (float)count;
            // conf.at<float>(i,j) = 1 - (average / response[0]);
        }
    }

    // Normalize
    // double minConf ;
    // double maxConf ;
    // minMaxLoc(conf, &minConf, &maxConf);

    // for(int k =0 ; k < h; k++)
    //     for(int z = 0; z < w; z++){
    //         conf.at<float>(k,z) = float(conf.at<float>(k,z) - minConf) / float(maxConf - minConf);
    //         cout << conf.at<float>(k,z) << endl;
    //     }
}

void RoadDetector::drawOrientation(int grid, int lenLine)
{
    int maxH = percImgDetect * h;
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

void RoadDetector::drawConfidence()
{
    Mat F = image.clone();
    for(int i = 0 ; i < h ; i++)
    {
        for(int j = 0 ; j < w; j++)
        {
            if(conf.at<float>(i,j) > 0.4){
                F.at<cv::Vec3b>(i,j)[0] = 255;
                F.at<cv::Vec3b>(i,j)[1] = 0;
                F.at<cv::Vec3b>(i,j)[2] = 255;
            }
        }
    }
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

// float RoadDetector::getScoreOrientation2(Point p, float angle, int thr)
// {
//     int step = 3;
//     float xStep = step * cos(angle);
//     float yStep = step * sin(angle);
//     int gap = h / (step * 16);
//     int total = 0;
//     int cor = 0;
//     int difSum = 0;
//     float radius = 0.35 * h;

//     float x = p.x - gap * xStep;
//     float y = p.y + gap * yStep;

//     while(x > 0 && x < w && y > 0 && y < h)
//     {
//         total += 1;
//         double dif = theta.at<float>(int(y), int(x)) - angle;
//         difSum += dif;
//         if(abs(dif) <= thr)
//             cor += 1;
//         x -= xStep;
//         y += yStep;
//     }

//     if(total < int(0.3 * h / step))
//         return -total;
//     else
//         return cor / sqrt(total);
// }

float RoadDetector::getScore(Point point, float threshold)
{
    vector<float> scores(numOrientations);

    for(int i = 0; i < numOrientations; i++)
    {
        scores[i] = getScoreOrientation(point, orientations[i], threshold);
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
  float result  = atan2f(v2y,v2x) - atan2f(v1y,v1x);
  return result;
}

float RoadDetector::getScoreKong(int x,int y, float radius)
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
                if(xx == 0 && yy == 0)
                    continue;

                // SOLVE THIS. DISTANCE AND ANGLE TOO BIG 

                float distance = sqrt(xx*xx + yy*yy);
                //cout << "Radius: " << radius << endl;
                //cout << "xx: " << xx << " yy " << yy << " distance: " << distance << endl;
                float gaborAngle = theta.at<float>(y+yy,x+xx); 
                float pvoAngle = abs(getAngle(x,y,x+xx,y+yy,gaborAngle));
                //cout << "Angle: " << pvoAngle << " Gabor angle: " << gaborAngle << endl;
                //float pvoAngle = abs(angle - gaborAngle);
                float pvoAngleDeg = pvoAngle * 180.0 / CV_PI ;
                float kongsAngleThreshold = 5/(1+2*distance);
                cout << "kongsAngleThreshold: " << kongsAngleThreshold << endl;
                cout << "pvoAngleDeg: " << pvoAngleDeg << endl;
                float kongsScore = 1/(1+((pvoAngleDeg*distance)*(pvoAngleDeg*distance)));
                if(pvoAngleDeg <= kongsAngleThreshold || pvoAngleDeg <= 3){
                    score+= kongsScore;
                }
            }
        }
    }
    return score;
}

Point RoadDetector::findVanishingPoint(int type, int n)
{
    double threshold = 1 / (2*CV_PI);
    votes = Mat::zeros(h, w, CV_32F);
    double radius = 0.35 * sqrt(w*w + h*h);
    int maxH = percImgDetect * h;

    clock_t start;
    cout << "Starting voter to find vanishing point" << endl;
    start = std::clock();

    for (int i = margin_h; i < maxH; i++)
  {
    for (int j = margin_w ; j < w - margin_w; j++)
        {
            //if(conf.at<float>(i,j) > 0.3){
            //if((i % pixelGrid == 0 && j % pixelGrid == 0)){
                if(type ==0 )
                    votes.at<float>(i,j) = getScore(Point(j,i), threshold);
                else
                    votes.at<float>(i,j) = getScoreKong(j,i, radius);
            //}
        }
    }

    double duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;
    cout << "voter time " << duration << endl;

    double min_val, max_val;
    Point min_p, max_p;
    minMaxLoc(votes, &min_val, &max_val, &min_p, &max_p);
    circle(image, max_p, 3, Scalar(255, 255, 0), -1);

    return max_p;
}

float distancePoints(Point a, Point b)
{
    float difx = (a.x - b.x);
    float dify = (a.y - b.y);
    return sqrt(difx*difx  + dify*dify);
}

bool is_between(Point a, Point c, Point b)
{
    float dif = distancePoints(a,c) + distancePoints(c,b) - distancePoints(a,b);
    return (dif <= 0.001 && dif >= -0.001);
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
    int x1  = p.x + int( diag * cos(angle));
    int y1 = p.y - int( diag * sin(angle));

    //circle(image, Point(x1,y1), 4, Scalar(255, 255, 255), -1);
    //circle(image, p, 4, Scalar(255, 255, 0), -1);
    // if(angle < CV_PI && angle > 0.0)
    // {
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
            for(int j = margin_w ; j < w -margin_w ; j++)
            {    
                if(is_between(p, Point(j,i), Point(x1,y1))){
                    //float dpv = ptr_dist[abs(j - p.y)];
                    distance = distancePoints(p, Point(j,i)) / maxDistance;
                    //distance = dpv / maxDistance;
                    distanceFunction = exp(-(distance * distance) / (2.0 * variance ) );
                    votes.at<float>(i,j) += (sinTheta) * distanceFunction;
                    //votes.at<float>(i,j) = votes.at<float>(i,j) +  1;
                    //votes.at<float>(i,j) += (sinTheta);
                    //votes.at<float>(i,j) += 1;
                }
            }
        }
    // }
}

Point RoadDetector::findVanishingPoint2()
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

    for (int i = margin_h ; i < maxH ; i++)
    {
        for (int j = margin_w  ; j < w - margin_w ; j++)
        {
            voter(Point(j,i), votes, dist_table);
            //voter(Point(200,250), votes, dist_table);
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

    circle(image, max_p, 6, Scalar(255, 255, 0), -1);
    
    return max_p;
}

void drawArrow(Mat &img, const Point &p_start, double theta, 
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

void RoadDetector::direction(Point van_point)
{
    double theta = atan2(h - van_point.y, van_point.x - w / 2);
    drawArrow(image, Point(w / 2, h), theta);
}

void RoadDetector::drawLines(const Point &van_point, const vector<float> &line_angles)
{
    int diag = h + w;
    for (int i = 0; i < line_angles.size(); ++i)
    {
        int temp_x = van_point.x - diag * cos(line_angles[i]);
        int temp_y = van_point.y + diag * sin(line_angles[i]);
        line(image, van_point, Point(temp_x, temp_y), Scalar(0, 0, 255), 3);
    }
    circle(image, van_point, 6, Scalar(0, 255, 255), -1);
}

// void RoadDetector::widthRoad(Point van_point)
// {
//     float angle = theta.at<float>(van_point);
//     float old_angle = angle;
//     float current_angle;
//     int count =0 ;
//     for(int i = van_point.x - 1; i > 0 ; i--)
//     {
//         count += 1;
//         float current_angle = theta.at<float>(Point(i,van_point.y));
//         cout << old_angle - current_angle << endl;
//         if(abs(old_angle - current_angle) < 0.2){
//             count = 0;
//             circle(image, Point(i,van_point.y), 4, Scalar(255, 255, 255), -1);
//         }
//         if(count > 20)
//             break;
//         cout << count << endl;

//         //old_angle = current_angle;
//     }

//     count = 0;
//     for(int i = van_point.x + 1; i < w ; i++)
//     {
//         count += 1;
//         float current_angle = theta.at<float>(Point(i,van_point.y));
//         cout << old_angle - current_angle << endl;
//         if(abs(old_angle - current_angle) < 0.2){
//             count = 0;
//             circle(image, Point(i,van_point.y), 4, Scalar(255, 255, 255), -1);
//         }
//         if(count > 20)
//             break;
//         cout << count << endl;
//         //old_angle = current_angle;
//     }
// }

void RoadDetector::widthRoad(Point van_point)
{
    float angle = theta.at<float>(van_point);
    int i = van_point.x;
    int j = van_point.y;
    int maxH = percImgDetect * h;
    while(i < w - margin_w && j < maxH)
    {
        float current_angle = theta.at<float>(Point(i,van_point.y));
        cout << abs(angle - current_angle) << endl;
        if(abs(angle - current_angle) < 3){
            circle(image, Point(i,j), 4, Scalar(255, 255, 255), -1);
        }
        i += 1;
        j += 1;
    }
}

Point RoadDetector::computeBottomPoint(Point& point,float angle)
{

  if(angle < 0){
        angle += 180;
    }
    if(angle == 0.0){
        angle = 5.0;
    }
    int x1 = point.x, y1 = point.y, y2 = h;
    float x2 = ((y2 - y1) / tan((angle/180.0)*CV_PI))+x1;
    if(x2 > w)
        x2 = w;

    return Point(x2,y2);
    
}

float RoadDetector::computeOCR(Point& vanishing_point, float angle)
{
    Point p1 = vanishing_point;
    Point p2 = computeBottomPoint(p1, angle);

    int dx = p2.x - p1.x, ax = abs(dx) << 1, sx = signOf(dx);
    int dy = p2.y - p1.y, ay = abs(dy) << 1, sy = signOf(dy);
    int x = vanishing_point.x;
    int y = vanishing_point.y;

    int count = 0;
    int point = 0;
    float angleRad = (angle/180.0)*CV_PI;

    if(ax > ay)
    {
        int d = ay - (ax >> 1);
        while(x != p2.x)
        {
            if(insideImage(x,y,h,w))
            {
                if(theta.at<float>(x,y) == angleRad)
                    point++;
                count++;
            }
            if(d >= 0)
            {
                y += sy;
                d -= ax;
            }
            x += sx;
            d += ay;
        }
    }
    else
    {
        int d = ax - (ay >> 1);
        while(y != p2.y)
        {
            if(insideImage(x,y,h,w))
            {
                if(theta.at<float>(x,y) == angle)
                    point++;
                count++;
            }
            if(d >= 0)
            {
                x += sx;
                d -= ay;
            }
            y += sy;
            d += ax;
        }
    }

    if(count == 0)
        return -1;
    return (float)point / (float)count ;
}

void RoadDetector::detectEdges(Point& vanishing_point)
{
    vector<float> scoresOCR(36);
    vector<float> angles(36);
    float angle = 0;
    for(int i = 0; i < numOrientations; i++){
        scoresOCR[i] = computeOCR(vanishing_point,orientations[i]);
        cout << scoresOCR[i] << endl;
    }
    int index = max_element(scoresOCR.begin(), scoresOCR.end()) - scoresOCR.begin();
    float max_score = angles[index], score_thr = max_score * 0.3f;
    cout << max_score << endl;
    Point p2 = computeBottomPoint(vanishing_point, max_score);
    line(image, vanishing_point, p2, Scalar(255,100,40), 2);
}

void RoadDetector::findVanishingPointMistry()
{
    Mat detected_edges;
    Mat src, dst;
    int edgeThresh = 1;
    int lowThreshold = 50;
    int const max_lowThreshold = 100;
    int ratio = 3;
    int kernel_size = 3;
    char* window_name = "Edge Map";

    /// Reduce noise with a kernel 3x3
    blur( imageGray, detected_edges, Size(3,3) );

    /// Canny detector
    Canny( detected_edges, detected_edges, lowThreshold, lowThreshold*ratio, kernel_size );

    /// Using Canny's output as a mask, we display our result
    dst = Scalar::all(0);

    image.copyTo( dst, detected_edges);
    imshow( window_name, dst );
    waitKey(0);
}
