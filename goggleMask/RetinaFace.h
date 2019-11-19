#ifndef RETINAFACE_H
#define RETINAFACE_H

#include <iostream>
#include <vector>
#include <map>
#include <opencv2/opencv.hpp>
#include "tensorrt/trtretinafacenet.h"
#include "goggleMaskClassify.h"
#include "goggleMaskAPI.h"

using namespace cv;
using namespace std;

struct anchor_win
{
    float x_ctr;
    float y_ctr;
    float w;
    float h;
};

//struct anchor_box
//{
//    float x1;
//    float y1;
//    float x2;
//    float y2;
//};
//
//struct FacePts
//{
//    float x[5];
//    float y[5];
//};
//
//struct FaceDetectInfo
//{
//    float score;
//    anchor_box rect;
//    FacePts pts;
//};

struct anchor_cfg
{
public:
    int STRIDE;
    vector<int> SCALES;
    int BASE_SIZE;
    vector<float> RATIOS;
    int ALLOWED_BORDER;

    anchor_cfg()
    {
        STRIDE = 0;
        SCALES.clear();
        BASE_SIZE = 0;
        RATIOS.clear();
        ALLOWED_BORDER = 0;
    }
};

class RetinaFace : GoggleMaskApi
{
public:
    RetinaFace();
    ~RetinaFace();

    void detectBatchImages(vector<cv::Mat> imgs, float threshold, vector<float> &scales, vector<vector<FaceDetectInfo>> &faceInfos);
    void detect(Mat &img, float threshold, float &scale, vector<FaceDetectInfo> &faceInfo);

    void InitFaceDetector(string modelRootPath, string network = "net3", float nms = 0.4);
    void InitFaceClassifier(string modelRootPath, string network="goggleMask");
    bool isInitFaceDetector() { return isInitDetector; }
    bool isInitFaceClassifier() { return isInitClassifier; }
    void detectFaces(std::string imgPath, float threshold, float &scale, vector<FaceDetectInfo> &faceInfo);
    void detectFaces(cv::Mat img, float threshold, float &scale, vector<FaceDetectInfo> &faceInfo);
    void classifyFace(std::string faceImgPath, std::vector<float> &outputs);
    void classifyFace(cv::Mat faceImg, std::vector<float> &outputs);

private:
    anchor_box bbox_pred(anchor_box anchor, cv::Vec4f regress);
    vector<anchor_box> bbox_pred(vector<anchor_box> anchors, vector<cv::Vec4f> regress);
    vector<FacePts> landmark_pred(vector<anchor_box> anchors, vector<FacePts> facePts);
    FacePts landmark_pred(anchor_box anchor, FacePts facePt);
    static bool CompareBBox(const FaceDetectInfo &a, const FaceDetectInfo &b);
    std::vector<FaceDetectInfo> nms(std::vector<FaceDetectInfo> &bboxes, float threshold);
    void pre_process(Mat img, int inputW, int inputH, float &scale, Mat &resize);
private:

    bool isInitDetector = false;
    bool isInitClassifier = false;

    TrtRetinaFaceNet *detectTrtNet = nullptr;
    GoggleMaskClassify *goggleMaskClassify = nullptr;
    float *cpuBuffers;

    float pixel_means[3] = {0.0, 0.0, 0.0};
    float pixel_stds[3] = {1.0, 1.0, 1.0};
    float pixel_scale = 1.0;

    int ctx_id;
    string network;
    float decay4;
    float nms_threshold;
    bool vote;
    bool nocrop;

    vector<float> _ratio;
    vector<anchor_cfg> cfg;

    vector<int> _feat_stride_fpn;
    //每一层fpn的anchor形状
    map<string, vector<anchor_box>> _anchors_fpn;
    //每一层所有点的anchor
    map<string, vector<anchor_box>> _anchors;
    //每一层fpn有几种形状的anchor
    //也就是ratio个数乘以scales个数
    map<string, int> _num_anchors;

};

#endif // RETINAFACE_H
