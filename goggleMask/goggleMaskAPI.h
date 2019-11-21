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

struct GoggleDetectInfo {
    float score;
    int label;             // 0--goggleMask, 1--goggle, 2--mask, 3--no
    float x1, y1, x2, y2;  // face box : left-top point and right-bottom point
};

class GoggleMaskApi {

protected:
    virtual ~GoggleMaskApi() {};

public:
    static GoggleMaskApi *CreateGMApi();
    static void DestroyGMApi(GoggleMaskApi **api);

    /**
     * init face detect model
     * @param modelRootPath : model root path, should contain "mxnet.prototxt" and "mxnet.caffemodel"
     * @param network : network type, default is "net3"
     * @param nms : nms threshold
     */
    virtual void InitFaceDetector(string modelRootPath, string network = "net3", float nms = 0.4) = 0;

    /**
     * init face classify model
     * @param modelRootPath : model root path, should contain "goggle_classify.prototxt" and "goggle_classify.caffemodel"
     * @param network : network name, you can ignore.
     */
    virtual void InitFaceClassifier(string modelRootPath, string network="goggleMask") = 0;

    /**
     * whether face detect model inited
     * @return bool, inited model or not
     */
    virtual bool isInitFaceDetector() = 0;

    /**
     * whether face classify model inited
     * @return bool, inited model or not
     */
    virtual bool isInitFaceClassifier() = 0;

    /**
     * detect faces in the image
     * @param imgPath : input image path
     * @param threshold : face threshold
     * @param scale : face scale
     * @param faceInfo : detect result, face box(anchor_box) need to multiply scale to get the exact box
     */
    virtual void detectFaces(std::string imgPath, float threshold, float &scale, vector<FaceDetectInfo> &faceInfo) = 0;
    virtual void detectFaces(cv::Mat img, float threshold, float &scale, vector<FaceDetectInfo> &faceInfo) = 0;

    /**
     * classify one face image
     * @param faceImgPath : input face image path
     * @param outputs : classify result scores, size(4), 0--goggle_mask, 1--goggle, 2--mask, 3--no
     */
    virtual void classifyFace(std::string faceImgPath, std::vector<float> &outputs) = 0;
    virtual void classifyFace(cv::Mat faceImg, std::vector<float> &outputs) = 0;


    /**
     * detect goggle_mask face
     * @param faceImgPath : input face image path
     * @param outputs : detect result
     * @param faceThreshold : detect face threshold, default is 0.5
     */
    virtual void detectGoggle(std::string imgPath, std::vector<GoggleDetectInfo> &outputs, float faceThreshold = 0.5) = 0;
    virtual void detectGoggle(cv::Mat img, std::vector<GoggleDetectInfo> &outputs, float faceThreshold = 0.5) = 0;

};

#endif
