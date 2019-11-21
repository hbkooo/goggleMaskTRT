#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include "goggleMaskAPI.h"
using namespace cv;
using namespace std;

int main(int argc, char** argv) {

    std::vector<string> labels = {"goggleMask", "goggle", "mask", "no"};
    std::vector<Scalar> colors = {Scalar(0,0,255), Scalar(0,255,0),
                                  Scalar(255,0,200), Scalar(200,255,0)};

    string path = "../../model";
    string imgPath = "../../img/img.jpg";

    if(argc == 2) imgPath = argv[1];

    std::cout << "detect image : " << imgPath << endl;

    GoggleMaskApi * goggleMaskApi = GoggleMaskApi::CreateGMApi();

    if(!goggleMaskApi->isInitFaceDetector()) {
        goggleMaskApi->InitFaceDetector(path);
    }
    if(!goggleMaskApi->isInitFaceClassifier()) {
        goggleMaskApi->InitFaceClassifier(path);
    }

    std::vector<GoggleDetectInfo> outputs;
    cv::Mat img = cv::imread(imgPath);
    goggleMaskApi->detectGoggle(img, outputs);

    for(int i = 0; i < outputs.size(); i++) {
        GoggleDetectInfo goggleDetectInfo = outputs[i];
        cv::rectangle(img, cv::Point(goggleDetectInfo.x1, goggleDetectInfo.y1),
                cv::Point(goggleDetectInfo.x2, goggleDetectInfo.y2),
                colors[goggleDetectInfo.label], 2);
        cv::putText(img, labels[goggleDetectInfo.label],
                cv::Point((goggleDetectInfo.x1 + goggleDetectInfo.x2)/2,(goggleDetectInfo.y1 + goggleDetectInfo.y2)/2),
                FONT_HERSHEY_SIMPLEX,1,colors[goggleDetectInfo.label],2,5);
    }

    std::cout << "save result : ./result.jpg\n";
    imwrite("./result.jpg", img);

    GoggleMaskApi::DestroyGMApi(&goggleMaskApi);

    return 0;
}

