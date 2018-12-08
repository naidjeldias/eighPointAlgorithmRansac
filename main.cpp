#include <iostream>
#include <opencv2/opencv.hpp>
#include "eightpoint.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"

using namespace cv;

int main() {

    EightPoint eightPoint;

    std::vector<KeyPoint> kpts_l, kpts_r;
    Mat descpt_l, descpt_r, image_mathes;
    std::vector< DMatch > matches;
    //loading images
    Mat left_frame = imread("/media/nigel/Dados/Documents/Projetos/CLionProjects/RANSAC/left1.png");
    Mat right_frame = imread("/media/nigel/Dados/Documents/Projetos/CLionProjects/RANSAC/right1.png", 0);

    cvtColor(left_frame, left_frame, COLOR_RGB2GRAY);
//    cvtColor(right_frame, right_frame, COLOR_RGB2GRAY);

    Ptr<FeatureDetector> detector = ORB::create();

    detector -> detectAndCompute(left_frame, Mat(), kpts_l, descpt_l);
    detector -> detectAndCompute(right_frame, Mat(), kpts_r, descpt_r);

    if(descpt_l.type() != CV_32F){
        descpt_l.convertTo(descpt_l, CV_32F);
    }
    if(descpt_r.type() != CV_32F){
        descpt_r.convertTo(descpt_r, CV_32F);
    }

    FlannBasedMatcher matcher;
    matcher.match(descpt_l, descpt_r, matches );

    std::vector<DMatch>     finalMatches;
    std::vector<bool>       inliers;
    cv::Mat                 fmat;
    cv::Mat                 epLinesLeft;
    cv::Mat                 epLinesRight;
    cv::Mat                 finalMatchImage;

    eightPoint.setRansacParameters(0.99, 8, 100, 0.05);
    fmat = eightPoint.ransacEightPointAlgorithm(matches, kpts_l, kpts_r, finalMatches, inliers, false);

//    eightPoint.setRansacParameters(0.99, 8, 100, 0.09);
//    fmat = eightPoint.ransacEightPointAlgorithm(matches, kpts_l, kpts_r, finalMatches, inliers, true);

    //std::cout << "Passou" << std::endl;
    std::cout << fmat << std::endl;
    epLinesLeft     = eightPoint.drawEpLines(kpts_l, kpts_r, matches, fmat, inliers, 0, left_frame);
    epLinesRight    = eightPoint.drawEpLines(kpts_l, kpts_r, matches, fmat, inliers, 1, right_frame);

    drawMatches(left_frame, kpts_l, right_frame, kpts_r, matches, image_mathes, Scalar::all(-1), Scalar::all(-1),
                std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    drawMatches(left_frame, kpts_l, right_frame, kpts_r, finalMatches, finalMatchImage, Scalar::all(-1), Scalar::all(-1),
                std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    imshow("matches", image_mathes);
    imshow("Final matches", finalMatchImage);
    imshow("Epipole lines left", epLinesLeft);
    imshow("Epipole lines Right", epLinesRight);

    waitKey(0);

    return 0;
}