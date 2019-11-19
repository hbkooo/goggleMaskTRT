#ifndef GOGGLEMASKAPI_H
#define GOGGLEMASKAPI_H

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
using namespace std;

struct anchor_box
{
    float x1;
    float y1;
    float x2;
    float y2;
};

struct FacePts
{
    float x[5];
    float y[5];
};

struct FaceDetectInfo
{
    float score;
    anchor_box rect;
    FacePts pts;
};

class GoggleMaskApi {

protected:
    virtual ~GoggleMaskApi() {};

public:
    static GoggleMaskApi *CreateGMApi();
    static void DestroyGMApi(GoggleMaskApi **api);

    virtual void InitFaceDetector(const string modelRootPath, string network = "net3", float nms = 0.4) = 0;
    virtual void InitFaceClassifier(const string modelRootPath, string network="goggleMask") = 0;

    virtual bool isInitFaceDetector() = 0;

    virtual bool isInitFaceClassifier() = 0;

    /**
     * detect faces in the image
     * @param imgPath : input image path
     * @param threshold : face threshold
     * @param scale : face scale
     * @param faceInfo : detect result, face box(anchor_box) need to multiply scale to get the exact box
     */
    virtual void detectFaces(const std::string imgPath, float threshold, float &scale, vector<FaceDetectInfo> &faceInfo) = 0;
    virtual void detectFaces(cv::Mat img, float threshold, float &scale, vector<FaceDetectInfo> &faceInfo) = 0;

    /**
     * classify one face image
     * @param faceImgPath : input face image path
     * @param outputs : classify result scores, size(4), 0--goggle_mask, 1--goggle, 2--mask, 3--no
     */
    virtual void classifyFace(const std::string faceImgPath, std::vector<float> &outputs) = 0;
    virtual void classifyFace(cv::Mat faceImg, std::vector<float> &outputs) = 0;

};

#endif