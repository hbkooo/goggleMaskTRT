#include "goggleMaskClassify.h"
#include <cuda_runtime_api.h>

void imageROIResize8U3C(void *src, int srcWidth, int srcHeight, cv::Rect imgROI, void *dst, int dstWidth, int dstHeight);
void convertBGR2RGBfloat(void *src, void *dst, int width, int height, cudaStream_t stream);
void imageSplit(const void *src, float *dst, int width, int height, cudaStream_t stream);

//######################################################################
// classify
//######################################################################

GoggleMaskClassify::GoggleMaskClassify(string &model, string network)
    : network(network)
{
    trtNet = new TrtGoggleMaskNet("goggleClassify");
    trtNet->buildTrtContext(model + "/goggle_classify.prototxt", model + "/goggle_classify.caffemodel",
            model + "/goggle_classify.engine");

    int maxbatchsize = trtNet->getMaxBatchSize();
    int channels = trtNet->getChannel();
    int inputW = trtNet->getNetWidth();
    int inputH = trtNet->getNetHeight();
    //
    int inputsize = maxbatchsize * channels * inputW * inputH * sizeof(float);
    cpuBuffers = (float *)malloc(inputsize);
    memset(cpuBuffers, 0, inputsize);
}

GoggleMaskClassify::~GoggleMaskClassify()
{
    delete trtNet;
    free(cpuBuffers);
}

void GoggleMaskClassify::pre_process(Mat img, int inputW, int inputH, float &scale, Mat &resize){
    float sw = 1.0 * img.cols / inputW;
    float sh = 1.0 * img.rows / inputH;
    scale = sw > sh ? sw : sh;
    scale = scale > 1.0 ? scale : 1.0;
//	std::cout << "\n\n\n\n\n\n\n\n\n\n\nloading...   \n\n\n\n\nimg.scale : " << scale << std::endl;
    if (scale > 1)
    {
        if (sw > sh)
        {
            cv::resize(img, resize, cv::Size(), 1 / scale, 1 / scale);
            cv::copyMakeBorder(resize, resize, 0, inputH - resize.rows, 0, 0, cv::BORDER_CONSTANT, cv::Scalar(0));
        }
        else
        {
            cv::resize(img, resize, cv::Size(), 1 / scale, 1 / scale);
            cv::copyMakeBorder(resize, resize, 0, 0, 0, inputW - resize.cols, cv::BORDER_CONSTANT, cv::Scalar(0));
        }
    }
    else
    {
        //直接补边到目标大小
        cv::copyMakeBorder(img, resize, 0, inputH - img.rows, 0, inputW - img.cols, cv::BORDER_CONSTANT, cv::Scalar(0));
    }
}

void GoggleMaskClassify::classify(Mat &img, std::vector<float> &results, float &scale)
{
    if (img.empty())
    {
        return;
    }
    int inputW = trtNet->getNetWidth();
    int inputH = trtNet->getNetHeight();
    int inputChannels = trtNet->getChannel();

    cv::Mat resize;
//    pre_process(img.clone(), inputW, inputH, scale, resize);
    cv::resize(img.clone(), resize, cv::Size(inputW, inputH));

//    cv::Mat face = imread(argv[2]);
//    cv::resize(face.clone(), face, cv::Size(32, 32));
//    imwrite("resized.jpg", face);

    resize.convertTo(resize, CV_32FC3);
//    cvtColor(resize, resize, CV_BGR2RGB);
//    std::cout << "resize.shape( " << resize.cols << ", " << resize.rows << " )" << std::endl;

    vector<Mat> input_channels;
    float *input_data = cpuBuffers;

    for (int i = 0; i < trtNet->getChannel(); ++i)
    {
        Mat channel(inputH, inputW, CV_32FC1, input_data);
        input_channels.push_back(channel);
        input_data += inputW * inputH;
    }

    cv::Mat mean_ = SetMean("120,130,153",inputW, inputH, inputChannels);  // 104,117,123
    cv::subtract(resize, mean_, resize);

    split(resize, input_channels);
    float *inputData = (float *)trtNet->getBuffer(0);
    cudaMemcpy(inputData, cpuBuffers, inputW * inputH * 3 * sizeof(float), cudaMemcpyHostToDevice);

    float* output = (float *)malloc(4* sizeof(float));
    trtNet->doInference(cpuBuffers, output);

    for(int i = 0; i < 4; i++) {
        results.push_back(output[i]);
    }

//    std::cout << "classify result : \ngoggle_mask : " << output[0] << "\ngoggle : " << output[1]
//                << "\nmask : " << output[2] << "\nno : " << output[3] << std::endl;

}

cv::Mat GoggleMaskClassify::SetMean(const string& mean_value, int inputW, int inputH, int channels_) {
    //mean_value
    cv::Mat mean_;

    cv::Scalar channel_mean;
    stringstream ss(mean_value);
    vector<float> values;
    string item;
    while (getline(ss, item, ',')) {
        float value = std::atof(item.c_str());
        values.push_back(value);
    }

    std::vector<cv::Mat> channels;
    for (int i = 0; i < channels_; ++i) {
        /* Extract an individual channel. */
        cv::Mat channel(inputH, inputW, CV_32FC1,
                        cv::Scalar(values[i]));
        channels.push_back(channel);
    }
    cv::merge(channels, mean_);
    return mean_;
}
