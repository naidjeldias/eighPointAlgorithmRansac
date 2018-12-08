//
// Created by nigel on 06/12/18.
//

#ifndef RANSAC_EIGHTPOINT_H
#define RANSAC_EIGHTPOINT_H
#include<opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <algorithm>

using namespace cv;

class EightPoint{

public:

    EightPoint();
    //----------------Ransac Parameters
    std::vector<int> generateRandomIndices(int maxIndice, int vecSize);
    void setRansacParameters(double probability, int minSet, int maxIteration, double maxError);
    double ransacProb, ransacTh;
    int ransacMinSet, ransacMaxIt;

    cv::Mat ransacEightPointAlgorithm(std::vector<DMatch> matches, std::vector<KeyPoint> kpt_l, std::vector<KeyPoint> kpt_r, std::vector<DMatch> &finalMatches, std::vector<bool> &inliers2D, bool normalize);
    //using normalized 8-point algorithm
    cv::Mat computeFundamentalMatrix(std::vector<KeyPoint> kpt_l, std::vector<KeyPoint> kpt_r, std::vector<int> indices, cv::Mat leftScalingMat, cv::Mat rightScalingMat, std::vector<DMatch> matches, bool normalize);
    //normalize data before compute fundamental matrix - translation and scaling of each umage so that
    //the centroid of the reference points is at the origin of the coordinates and the RMS distance from the origin is equal to sqrt(2)
    void computeMatNormTransform(std::vector<KeyPoint> kpt_l, std::vector<KeyPoint> kpt_r, int nPts, cv::Mat &leftScalingMat, cv::Mat &rightScalingMat);
    double sampsonError(cv::Mat fmat, cv::Mat left_pt, cv::Mat right_pt);
    cv::Mat drawEpLines(std::vector<KeyPoint> pts_l, std::vector<KeyPoint> pts_r, std::vector<DMatch> matches, cv::Mat F, std::vector<bool> inliers, int rightFlag, cv::Mat image);

    void normalizeMatchesPointsCV(std::vector<Point2f> &pts_l, std::vector<Point2f> &pts_r, std::vector<DMatch> matches,
                                std::vector<KeyPoint> kpt_l, std::vector<KeyPoint> kpt_r, cv::Mat scalingLeft, cv::Mat scalingRight, std::vector<int> randval);

};



#endif //RANSAC_EIGHTPOINT_H
