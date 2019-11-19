#ifndef GOGGLEMASKCLASSIFY_H
#define GOGGLEMASKCLASSIFY_H

#include <iostream>
#include <vector>
#include <map>
#include <opencv2/opencv.hpp>
#include "tensorrt/trtgogglemask.h"

using namespace cv;
using namespace std;

class GoggleMaskClassify
{
public:
    GoggleMaskClassify(string &model, string network="goggleMask");
    ~GoggleMaskClassify();

    //void detectBatchImages(vector<cv::Mat> imgs, float threshold, vector<float> &scales, vector<vector<FaceDetectInfo>> &faceInfos);
    void classify(Mat &img, std::vector<float> &results, float &scale);

private:
    void pre_process(Mat img, int inputW, int inputH, float &scale, Mat &resize);
    cv::Mat SetMean(const string& mean_value, int inputW, int inputH, int channels_);
private:

    TrtGoggleMaskNet *trtNet;
    float *cpuBuffers;

    string network;

};

#endif // GOGGLEMASKCLASSIFY_H
