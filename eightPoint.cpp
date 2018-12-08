//
// Created by nigel on 06/12/18.
//

#include "eightpoint.hpp"

using namespace cv;

EightPoint::EightPoint(){
    //initialize random seed for ransac
    srand((unsigned)time(NULL));
}

void EightPoint::setRansacParameters(double probability, int minSet, int maxIteration, double maxError) {
    ransacMaxIt     = maxIteration;
    ransacMinSet    = minSet;
    ransacProb      = probability;
    ransacTh        = maxError;
}

std::vector<int> EightPoint::generateRandomIndices(int maxIndice, int vecSize){
    std::vector<int> randValues;
    int index;

//    do{
//        index = rand() % maxIndice;
//        if(!(std::find(randValues.begin(), randValues.end(), index) != randValues.end()))
//            randValues.push_back(index);
//    }while(randValues.size() < vecSize);

    do{
        index = rand() % maxIndice;

        randValues.push_back(index);
    }while(randValues.size() < vecSize);

//     std::cout << "Rand vector: ";
//     for(int i = 0; i < randValues.size(); i++)
//         std::cout << randValues.at(i) << std::endl;

    return randValues;
}

cv::Mat EightPoint::ransacEightPointAlgorithm(std::vector<DMatch> matches, std::vector<KeyPoint> kpt_l,
                                              std::vector<KeyPoint> kpt_r, std::vector<DMatch>& finalMatches, std::vector<bool> &bestInliers, bool normalize) {

    finalMatches.clear();

    std::vector<Point2f> pts_left, pts_right;
    //matrix for normalizing transformations
    Mat leftScaling     =  Mat::zeros(3,3,CV_64F);
    Mat rightScaling    =  Mat::zeros(3,3,CV_64F);
    //Fundamental matrix
    Mat fmat, bestFmat;


    if(normalize) computeMatNormTransform(kpt_l, kpt_r, kpt_l.size(), leftScaling, rightScaling);
    // normalizeMatchesPoints(pts_left, pts_right, matches, kpt_l, kpt_r, leftScaling, rightScaling);

    // std::vector<bool> bestInliers;
    int bestNumInliers = 8;

    int n = 0;
    long int r          = 1000;//adjusted dinamically
    while (n < r && n < ransacMaxIt){

        std::vector<int> randValues;    //vector of rand index
        std::vector<DMatch> matches_;   //inliers matches in each iteration
        //-----------find inliers
        std::vector<bool> inliers;
        int numInliers = 0;

        //compute indices to pick 8 random points
        randValues      = generateRandomIndices(matches.size(), ransacMinSet);

        //compute fundamental matrix with the subset
        fmat            = computeFundamentalMatrix(kpt_l, kpt_r, randValues, leftScaling, rightScaling, matches, normalize);
        //--------------------opencv method
        std::vector<Point2f>    pts_l, pts_r;
//        normalizeMatchesPointsCV(pts_l, pts_r, matches, kpt_l, kpt_r, leftScaling, rightScaling, randValues);
//        fmat            = findFundamentalMat(pts_l, pts_r, FM_8POINT);
//        std::cout << fmat << std::endl;


        if(fmat.empty()) continue;

        //validate model againts the init set
        for(int i = 0; i < matches.size() ; i++){

            //validadte against other elements
            if(!(std::find(randValues.begin(), randValues.end(), i) != randValues.end())){

                Mat X_l   = Mat::zeros(3,1,CV_64F);
                Mat X_r   = Mat::zeros(3,1,CV_64F);

                //point on left frame
                X_l.at<double>(0)     = kpt_l.at(matches[i].queryIdx).pt.x;
                X_l.at<double>(1)     = kpt_l.at(matches[i].queryIdx).pt.y;
                X_l.at<double>(2)     = 1.0;
                //point on right frame
                X_r.at<double>(0)   = kpt_r.at(matches[i].trainIdx).pt.x;
                X_r.at<double>(1)   = kpt_r.at(matches[i].trainIdx).pt.y;
                X_r.at<double>(2)   = 1.0;

                double d   =  sampsonError(fmat, X_l, X_r);
//                std::cout<< "Sampson error" << d << std::endl;

                if( d < ransacTh){
                    inliers.push_back(true);
                    numInliers++;
                    matches_.push_back(matches[i]);
                    // std::cout << "matches inliers: " << matches_.size() << std::endl;
//                    std::cout << "Distance: " << d << std::endl;

//                     std::cout << "Left point: " << X_l << std::endl;
//                     std::cout << "Right point: " << X_r << std::endl;
                }else{
                    inliers.push_back(false);
                }


            }else{
                inliers.push_back(true);//the points from subset are considered as inliers
                matches_.push_back(matches[i]);
            }

        }

        if(numInliers > bestNumInliers){
//            std::cout << "Num inliers: " << numInliers << std::endl;
            bestInliers     = inliers;
            finalMatches    = matches_;

            bestFmat        = fmat;

            bestNumInliers  = numInliers;

            //fraction of inliers in the set of points
            double w    = (double) bestNumInliers / (double) matches.size();
//            std::cout << "fraction of inliers: " << w << std::endl;
            //probability of not all N points are inliers
            //in each iteration we pick N points that are all inliers with probability w^N
            double p1   = 1 - pow(w, ransacMinSet);
            p1 = MAX(LDBL_MIN, p1);     // Avoid division by -Inf
            p1 = MIN(1-LDBL_MIN, p1);   // Avoid division by 0.
//            std::cout << "probability : " << p1 << std::endl;
            //probability of not all N points are inliers in r iterations is (1 - w^N)^r
            //the probability that in r iteration, at least once, all N points are inliers: p = 1-(1 - W^N)^r
            r = log(1 - ransacProb)/log(p1);
//            std::cout << "r estimated: " << r << std::endl;
        }
        n ++;
    }

    // std::cout << "Inliers 2d: " << bestNumInliers << std::endl;

     std::cout << "Number of iterations: " << n << std::endl;
     std::cout << "Best num of inliers: " << bestNumInliers <<std::endl;
    // std::cout << "Size inliers vec: " << finalMatches.size() << std::endl;

    //draw epipole lines on left image
    // drawEpLines (kpt_l, kpt_r, matches, bestFmat, bestInliers, 0,im0);
    //draw epipole lines on right image
    // drawEpLines (kpt_l, kpt_r, matches, bestFmat, bestInliers, 1,im1);

    return bestFmat;
}

cv::Mat EightPoint::computeFundamentalMatrix(std::vector<KeyPoint> kpt_l, std::vector<KeyPoint> kpt_r, std::vector<int> indices, cv::Mat T, cv::Mat T_l, std::vector<DMatch> matches, bool normalize){
    //square matrix 9x9
    Mat A   = Mat::zeros(9,9,CV_64F);

    Mat  D, U, Vt;

    //fill mat A
    for (int row = 0; row < 8; row++){

        int i = indices.at(row);
        double xLeft, yLeft, xRight, yRight;

        if(normalize){
            xLeft = kpt_l.at(matches[i].queryIdx).pt.x * T.at<double> (0,0) + T.at<double>(0,2);
            yLeft = kpt_l.at(matches[i].queryIdx).pt.y * T.at<double> (1,1) + T.at<double>(1,2);

            xRight = kpt_r.at(matches[i].trainIdx).pt.x * T_l.at<double> (0,0) + T.at<double>(0,2);
            yRight = kpt_r.at(matches[i].trainIdx).pt.y * T_l.at<double> (1,1) + T.at<double>(1,2);
        }else{
            xLeft = kpt_l.at(matches[i].queryIdx).pt.x;
            yLeft = kpt_l.at(matches[i].queryIdx).pt.y;

            xRight = kpt_r.at(matches[i].trainIdx).pt.x;
            yRight = kpt_r.at(matches[i].trainIdx).pt.y;
        }


        A.at<double>(row,0) = xLeft*xRight;
        A.at<double>(row,1) = xRight*yLeft;
        A.at<double>(row,2) = xRight;
        A.at<double>(row,3) = yRight*xLeft;
        A.at<double>(row,4) = yRight*yLeft;
        A.at<double>(row,5) = yRight;
        A.at<double>(row,6) = xLeft;
        A.at<double>(row,7) = yLeft;
        A.at<double>(row,8) = 1.0;

    }

    // std::cout << "Mat A: "<< A << std::endl;
    // -------------------------retrieve fundamental matrix from SVD of A
    // F = UDV_t
    SVD::compute(A,D,U,Vt);

    std::vector<double> fvec = Vt.row(8);

    Mat F = Mat::zeros(3, 3, CV_64F);

    for(int i=0; i < F.rows; i++){
        F.at<double>(i,0) = fvec.at(3*i);
        F.at<double>(i,1) = fvec.at(3*i + 1);
        F.at<double>(i,2) = fvec.at(3*i + 2);
    }

    //--------------------------force rank 2 of F
    //D_l = diag(r,s,t) so F_l = U_ldiag(r,s,0)Vt_l
    Mat F_l, D_l, U_l, Vt_l;
    SVD::compute(F,D_l,U_l,Vt_l);
    Mat diag    = Mat::zeros(3,3, CV_64F);
    diag.at<double>(0,0) = D_l.at<double>(0);
    diag.at<double>(1,1) = D_l.at<double>(1);
    // std::cout << "Diag: " << diag << std::endl;
    F_l = U_l * diag * Vt_l;

    //-------------------------denormalize F_l
    if(normalize) F_l =  T_l.t() * F_l * T;

    //std::cout << "Fmatrix: " << Fmatrix << std::endl;
    F_l = F_l/F_l.at<double>(2,2);

    return F_l;

}

void EightPoint::computeMatNormTransform(std::vector<KeyPoint> kpt_l, std::vector<KeyPoint> kpt_r, int nPts, cv::Mat &leftScalingMat, cv::Mat &rightScalingMat){

    Point2f meanLeft;
    Point2f meanRight;

    meanLeft.x  = 0.0; meanLeft.y   = 0.0;
    meanRight.x = 0.0; meanRight.y  = 0.0;

    for(int i = 0; i<nPts; i++){

        meanLeft.x      += kpt_l.at(i).pt.x;
        meanLeft.y      += kpt_l.at(i).pt.y;

        meanRight.x     += kpt_r.at(i).pt.x;
        meanRight.y     += kpt_r.at(i).pt.y;

    }

    //means
    meanLeft.x      /= (double) nPts;
    meanLeft.y      /= (double) nPts;
    meanRight.x     /= (double) nPts;
    meanRight.y     /= (double) nPts;

    double leftDst      = 0.0;
    double rightDst     = 0.0;

    //normalization of the points: distance from centroide
    for(int i = 0; i < nPts; i++){

        double deltaX   = meanLeft.x - kpt_l.at(i).pt.x;
        double deltaY   = meanLeft.y - kpt_l.at(i).pt.y;

        leftDst         += sqrt(deltaX*deltaX + deltaY*deltaY);

        deltaX           = meanRight.x - kpt_r.at(i).pt.x;
        deltaY           = meanRight.y - kpt_r.at(i).pt.y;

        rightDst        += sqrt(deltaX*deltaX + deltaY*deltaY);
    }

    //mean of distance
    leftDst     /= (double) nPts;
    rightDst    /= (double) nPts;

    //scale factor
    double scaleLeft    = sqrt(2.0)/leftDst;
    double scaleRight   = sqrt(2.0)/rightDst;

    leftScalingMat.at<double>(0,0)  = scaleLeft;
    leftScalingMat.at<double>(1,1)  = scaleLeft;
    leftScalingMat.at<double>(0,2)  = -scaleLeft*meanLeft.x;
    leftScalingMat.at<double>(1,2)  = -scaleLeft*meanLeft.y;
    leftScalingMat.at<double>(2,2)  = 1.0;

    rightScalingMat.at<double>(0,0)  = scaleRight;
    rightScalingMat.at<double>(1,1)  = scaleRight;
    rightScalingMat.at<double>(0,2)  = -scaleRight*meanRight.x;
    rightScalingMat.at<double>(1,2)  = -scaleRight*meanRight.y;
    rightScalingMat.at<double>(2,2)  = 1.0;

    // std::cout << "Scaling Mat Left: "   << leftScalingMat   << std::endl;
    // std::cout << "Scaling Mat Right: "  << rightScalingMat  << std::endl;

}


double EightPoint::sampsonError(cv::Mat fmat, cv::Mat left_pt, cv::Mat right_pt){

    Mat matTmp         = right_pt.t() * fmat;
    matTmp             = matTmp  * left_pt;

    Mat letfTmpMat     = fmat * left_pt;
    Mat rightTmpMat    = fmat.t() * right_pt;

    double num = (double) matTmp.at<double>(0,0);

    double den = letfTmpMat.at<double>(0)*letfTmpMat.at<double>(0) + letfTmpMat.at<double>(1)*letfTmpMat.at<double>(1) +
                 rightTmpMat.at<double>(0)*rightTmpMat.at<double>(0) + rightTmpMat.at<double>(1)*rightTmpMat.at<double>(1);

    return ((num*num)/den);
}

cv::Mat EightPoint::drawEpLines(std::vector<KeyPoint> pts_l, std::vector<KeyPoint> pts_r, std::vector<DMatch> matches, cv::Mat F, std::vector<bool> inliers, int rightFlag, cv::Mat image){

    Mat border = Mat::zeros(4,2,CV_64F);
    Mat X_l   = Mat::zeros(3,1,CV_64F);
    Mat X_r   = Mat::zeros(3,1,CV_64F);
    Mat eplines;

    int w = image.size().width;
    int h = image.size().height;

//    Mat rgb = image.clone();
    Mat rgb;
    cvtColor(image, rgb, COLOR_GRAY2BGR);
    int count = 0;
    // std::vector<Point2f> points;
    for(int i = 0; i < pts_l.size(); i++ ){
        if(inliers.at(i)){
            count ++;
            //point on left frame
            X_l.at<double>(0)     = pts_l.at(matches[i].queryIdx).pt.x;
            X_l.at<double>(1)     = pts_l.at(matches[i].queryIdx).pt.y;
            X_l.at<double>(2)     = 1.0;
            //point on right frame
            X_r.at<double>(0)   = pts_r.at(matches[i].trainIdx).pt.x;
            X_r.at<double>(1)   = pts_r.at(matches[i].trainIdx).pt.y;
            X_r.at<double>(2)   = 1.0;

            Mat ep_line;

            //if zero draw in left image else draw in right image
            if(rightFlag == 0){
                ep_line = F.t() * X_r;
            }else
                ep_line = F * X_l;

            //computing ep lines
            double a    =   ep_line.at<double>(0);
            double b    =   ep_line.at<double>(1);
            double c    =   ep_line.at<double>(2);

            //borders and epipolar line intersection points
            border.at<double>(0,0) = 0.0;           border.at<double>(0,1) = -c/b;          //left
            border.at<double>(1,0) = w;             border.at<double>(1,1) = (-c-a*w)/b;    //right
            border.at<double>(2,0) = -c/a;          border.at<double>(2,1) = 0.0;           //up
            border.at<double>(3,0) = (-c-b*h)/a;    border.at<double>(3,1) = h;             //down
            //points of epipolar lines
            std::vector<double> linePts;

            for(int i = 0; i < 4; i++){
                double x = border.at<double>(i,0);
                double y = border.at<double>(i,1);
                if( x>=0 && x<=w && y>=0 && y<=h){
                    linePts.push_back(x);
                    linePts.push_back(y);
                }
            }

            if(linePts.size()>=4){
                Scalar color (rand() % 255,rand() % 255,rand() % 255);
                Point x0(linePts.at(0), linePts.at(1));
                Point x1(linePts.at(2), linePts.at(3));
                line(rgb, x0, x1, color, 1);

                if(rightFlag == 0){
                    Point x(X_l.at<double>(0), X_l.at<double>(1));
                    circle(rgb,x, 5, color, -1);
                }else{
                    Point x(X_r.at<double>(0), X_r.at<double>(1));
                    circle(rgb,x, 5, color, -1);
                }

            }
        }
    }
//    std::cout << "Num of points and lines: " << count << std::endl;
    return rgb;
    // computeCorrespondEpilines(points,1,F, eplines);

}

void EightPoint::normalizeMatchesPointsCV(std::vector<Point2f> &pts_l, std::vector<Point2f> &pts_r,
                                          std::vector<DMatch> matches, std::vector<KeyPoint> kpt_l,
                                          std::vector<KeyPoint> kpt_r, cv::Mat scalingLeft, cv::Mat scalingRight,
                                          std::vector<int> randval) {

    // std::cout << "Matches size: "   << matches.size() << std::endl;
    // std::cout << "Left kpts size: "   << kpt_l.size()   << std::endl;
    // std::cout << "Right kpts size: "   << kpt_r.size()   << std::endl;

    // std::cout << "Scale mat left" << scalingLeft << std::endl;

    for(int i = 0; i < randval.size(); i++){

        // KeyPoint pt_l;
        // KeyPoint pt_r;

        Point2f pt_l;
        Point2f pt_r;

        int pos = randval.at(i);

//        float xLeft    = kpt_l[matches[pos].queryIdx].pt.x * scalingLeft.at<float>(0,0) + scalingLeft.at<float>(0,2);
//        float yLeft    = kpt_l[matches[pos].queryIdx].pt.y * scalingLeft.at<float>(1,1) + scalingLeft.at<float>(1,2);
//
//        float xRight   = kpt_r[matches[pos].trainIdx].pt.x * scalingRight.at<float>(0,0) + scalingRight.at<float>(0,2);
//        float yRight   = kpt_r[matches[pos].trainIdx].pt.y * scalingRight.at<float>(1,1) + scalingRight.at<float>(1,2);

        double xLeft    = kpt_l[matches[pos].queryIdx].pt.x ;
        double yLeft    = kpt_l[matches[pos].queryIdx].pt.y ;

        double xRight   = kpt_r[matches[pos].trainIdx].pt.x ;
        double yRight   = kpt_r[matches[pos].trainIdx].pt.y ;

        pt_l.x = xLeft;
        pt_l.y = yLeft;

        pt_r.x = xRight;
        pt_r.y = yRight;

        pts_l.push_back(pt_l);
        pts_r.push_back(pt_r);
    }
}