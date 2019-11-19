#include "RetinaFace.h"
#include "goggleMaskAPI.h"

#include <cuda_runtime_api.h>

void imageROIResize8U3C(void *src, int srcWidth, int srcHeight, cv::Rect imgROI, void *dst, int dstWidth, int dstHeight);
void convertBGR2RGBfloat(void *src, void *dst, int width, int height, cudaStream_t stream);
void imageSplit(const void *src, float *dst, int width, int height, cudaStream_t stream);

//processing
anchor_win _whctrs(anchor_box anchor)
{
    //Return width, height, x center, and y center for an anchor (window).
    anchor_win win;
    win.w = anchor.x2 - anchor.x1 + 1;
    win.h = anchor.y2 - anchor.y1 + 1;
    win.x_ctr = anchor.x1 + 0.5 * (win.w - 1);
    win.y_ctr = anchor.y1 + 0.5 * (win.h - 1);

    return win;
}

anchor_box _mkanchors(anchor_win win)
{
    //Given a vector of widths (ws) and heights (hs) around a center
    //(x_ctr, y_ctr), output a set of anchors (windows).
    anchor_box anchor;
    anchor.x1 = win.x_ctr - 0.5 * (win.w - 1);
    anchor.y1 = win.y_ctr - 0.5 * (win.h - 1);
    anchor.x2 = win.x_ctr + 0.5 * (win.w - 1);
    anchor.y2 = win.y_ctr + 0.5 * (win.h - 1);

    return anchor;
}

vector<anchor_box> _ratio_enum(anchor_box anchor, vector<float> ratios)
{
    //Enumerate a set of anchors for each aspect ratio wrt an anchor.
    vector<anchor_box> anchors;
    for (size_t i = 0; i < ratios.size(); i++)
    {
        anchor_win win = _whctrs(anchor);
        float size = win.w * win.h;
        float scale = size / ratios[i];

        win.w = std::round(sqrt(scale));
        win.h = std::round(win.w * ratios[i]);

        anchor_box tmp = _mkanchors(win);
        anchors.push_back(tmp);
    }

    return anchors;
}

vector<anchor_box> _scale_enum(anchor_box anchor, vector<int> scales)
{
    //Enumerate a set of anchors for each scale wrt an anchor.
    vector<anchor_box> anchors;
    for (size_t i = 0; i < scales.size(); i++)
    {
        anchor_win win = _whctrs(anchor);

        win.w = win.w * scales[i];
        win.h = win.h * scales[i];

        anchor_box tmp = _mkanchors(win);
        anchors.push_back(tmp);
    }

    return anchors;
}

vector<anchor_box> generate_anchors(int base_size = 16, vector<float> ratios = {0.5, 1, 2},
                                    vector<int> scales = {8, 64}, int stride = 16, bool dense_anchor = false)
{
    //Generate anchor (reference) windows by enumerating aspect ratios X
    //scales wrt a reference (0, 0, 15, 15) window.

    anchor_box base_anchor;
    base_anchor.x1 = 0;
    base_anchor.y1 = 0;
    base_anchor.x2 = base_size - 1;
    base_anchor.y2 = base_size - 1;

    vector<anchor_box> ratio_anchors;
    ratio_anchors = _ratio_enum(base_anchor, ratios);

    vector<anchor_box> anchors;
    for (size_t i = 0; i < ratio_anchors.size(); i++)
    {
        vector<anchor_box> tmp = _scale_enum(ratio_anchors[i], scales);
        anchors.insert(anchors.end(), tmp.begin(), tmp.end());
    }

    if (dense_anchor)
    {
        assert(stride % 2 == 0);
        vector<anchor_box> anchors2 = anchors;
        for (size_t i = 0; i < anchors2.size(); i++)
        {
            anchors2[i].x1 += stride / 2;
            anchors2[i].y1 += stride / 2;
            anchors2[i].x2 += stride / 2;
            anchors2[i].y2 += stride / 2;
        }
        anchors.insert(anchors.end(), anchors2.begin(), anchors2.end());
    }

    return anchors;
}

vector<vector<anchor_box>> generate_anchors_fpn(bool dense_anchor = false, vector<anchor_cfg> cfg = {})
{
    //Generate anchor (reference) windows by enumerating aspect ratios X
    //scales wrt a reference (0, 0, 15, 15) window.

    vector<vector<anchor_box>> anchors;
    for (size_t i = 0; i < cfg.size(); i++)
    {
        //stride从小到大[32 16 8]
        anchor_cfg tmp = cfg[i];
        int bs = tmp.BASE_SIZE;
        vector<float> ratios = tmp.RATIOS;
        vector<int> scales = tmp.SCALES;
        int stride = tmp.STRIDE;

        vector<anchor_box> r = generate_anchors(bs, ratios, scales, stride, dense_anchor);
        anchors.push_back(r);
    }

    return anchors;
}

vector<anchor_box> anchors_plane(int height, int width, int stride, vector<anchor_box> base_anchors)
{
    /*
    height: height of plane
    width:  width of plane
    stride: stride ot the original image
    anchors_base: a base set of anchors
    */

    vector<anchor_box> all_anchors;
    for (size_t k = 0; k < base_anchors.size(); k++)
    {
        for (int ih = 0; ih < height; ih++)
        {
            int sh = ih * stride;
            for (int iw = 0; iw < width; iw++)
            {
                int sw = iw * stride;

                anchor_box tmp;
                tmp.x1 = base_anchors[k].x1 + sw;
                tmp.y1 = base_anchors[k].y1 + sh;
                tmp.x2 = base_anchors[k].x2 + sw;
                tmp.y2 = base_anchors[k].y2 + sh;
                all_anchors.push_back(tmp);
            }
        }
    }

    return all_anchors;
}

void clip_boxes(vector<anchor_box> &boxes, int width, int height)
{
    //Clip boxes to image boundaries.
    for (size_t i = 0; i < boxes.size(); i++)
    {
        boxes[i].x1 = std::max<float>(std::min<float>(boxes[i].x1, width - 1), 0);
        boxes[i].y1 = std::max<float>(std::min<float>(boxes[i].y1, height - 1), 0);
        boxes[i].x2 = std::max<float>(std::min<float>(boxes[i].x2, width - 1), 0);
        boxes[i].y2 = std::max<float>(std::min<float>(boxes[i].y2, height - 1), 0);
    }
}

void clip_boxes(anchor_box &box, int width, int height)
{
    //Clip boxes to image boundaries.
    box.x1 = std::max<float>(std::min<float>(box.x1, width - 1), 0);
    box.y1 = std::max<float>(std::min<float>(box.y1, height - 1), 0);
    box.x2 = std::max<float>(std::min<float>(box.x2, width - 1), 0);
    box.y2 = std::max<float>(std::min<float>(box.y2, height - 1), 0);
}

//######################################################################
//retinaface
//######################################################################

RetinaFace::RetinaFace() { }

RetinaFace::~RetinaFace()
{
    std::cout << "delete space in ~RetinaFace\n";
    delete detectTrtNet;
    free(cpuBuffers);
    if(goggleMaskClassify != nullptr) delete goggleMaskClassify;
}

vector<anchor_box> RetinaFace::bbox_pred(vector<anchor_box> anchors, vector<cv::Vec4f> regress)
{
    //"""
    //  Transform the set of class-agnostic boxes into class-specific boxes
    //  by applying the predicted offsets (box_deltas)
    //  :param boxes: !important [N 4]
    //  :param box_deltas: [N, 4 * num_classes]
    //  :return: [N 4 * num_classes]
    //  """
    vector<anchor_box> rects(anchors.size());
    for (size_t i = 0; i < anchors.size(); i++)
    {
        float width = anchors[i].x2 - anchors[i].x1 + 1;
        float height = anchors[i].y2 - anchors[i].y1 + 1;
        float ctr_x = anchors[i].x1 + 0.5 * (width - 1.0);
        float ctr_y = anchors[i].y1 + 0.5 * (height - 1.0);

        float pred_ctr_x = regress[i][0] * width + ctr_x;
        float pred_ctr_y = regress[i][1] * height + ctr_y;
        float pred_w = exp(regress[i][2]) * width;
        float pred_h = exp(regress[i][3]) * height;

        rects[i].x1 = pred_ctr_x - 0.5 * (pred_w - 1.0);
        rects[i].y1 = pred_ctr_y - 0.5 * (pred_h - 1.0);
        rects[i].x2 = pred_ctr_x + 0.5 * (pred_w - 1.0);
        rects[i].y2 = pred_ctr_y + 0.5 * (pred_h - 1.0);
    }

    return rects;
}

anchor_box RetinaFace::bbox_pred(anchor_box anchor, cv::Vec4f regress)
{
    anchor_box rect;

    float width = anchor.x2 - anchor.x1 + 1;
    float height = anchor.y2 - anchor.y1 + 1;
    float ctr_x = anchor.x1 + 0.5 * (width - 1.0);
    float ctr_y = anchor.y1 + 0.5 * (height - 1.0);

    float pred_ctr_x = regress[0] * width + ctr_x;
    float pred_ctr_y = regress[1] * height + ctr_y;
    float pred_w = exp(regress[2]) * width;
    float pred_h = exp(regress[3]) * height;

    rect.x1 = pred_ctr_x - 0.5 * (pred_w - 1.0);
    rect.y1 = pred_ctr_y - 0.5 * (pred_h - 1.0);
    rect.x2 = pred_ctr_x + 0.5 * (pred_w - 1.0);
    rect.y2 = pred_ctr_y + 0.5 * (pred_h - 1.0);

    return rect;
}

vector<FacePts> RetinaFace::landmark_pred(vector<anchor_box> anchors, vector<FacePts> facePts)
{
    vector<FacePts> pts(anchors.size());
    for (size_t i = 0; i < anchors.size(); i++)
    {
        float width = anchors[i].x2 - anchors[i].x1 + 1;
        float height = anchors[i].y2 - anchors[i].y1 + 1;
        float ctr_x = anchors[i].x1 + 0.5 * (width - 1.0);
        float ctr_y = anchors[i].y1 + 0.5 * (height - 1.0);

        for (size_t j = 0; j < 5; j++)
        {
            pts[i].x[j] = facePts[i].x[j] * width + ctr_x;
            pts[i].y[j] = facePts[i].y[j] * height + ctr_y;
        }
    }

    return pts;
}

FacePts RetinaFace::landmark_pred(anchor_box anchor, FacePts facePt)
{
    FacePts pt;
    float width = anchor.x2 - anchor.x1 + 1;
    float height = anchor.y2 - anchor.y1 + 1;
    float ctr_x = anchor.x1 + 0.5 * (width - 1.0);
    float ctr_y = anchor.y1 + 0.5 * (height - 1.0);

    for (size_t j = 0; j < 5; j++)
    {
        pt.x[j] = facePt.x[j] * width + ctr_x;
        pt.y[j] = facePt.y[j] * height + ctr_y;
    }

    return pt;
}

bool RetinaFace::CompareBBox(const FaceDetectInfo &a, const FaceDetectInfo &b)
{
    return a.score > b.score;
}

std::vector<FaceDetectInfo> RetinaFace::nms(std::vector<FaceDetectInfo> &bboxes, float threshold)
{
    std::vector<FaceDetectInfo> bboxes_nms;
    std::sort(bboxes.begin(), bboxes.end(), CompareBBox);
    std::vector<size_t> idx(bboxes.size());
    for (unsigned i = 0; i < idx.size(); i++)
    {
        idx[i] = i;
    }
    while (idx.size() > 0)
    {
        int good_idx = idx[0];
        bboxes_nms.push_back(bboxes[good_idx]);

        std::vector<size_t> tmp = idx;
        idx.clear();

        for (unsigned i = 1; i < tmp.size(); i++)
        {
            int tmp_i = tmp[i];
            float inter_x1 = std::max(bboxes[good_idx].rect.x1, bboxes[tmp_i].rect.x1);
            float inter_y1 = std::max(bboxes[good_idx].rect.y1, bboxes[tmp_i].rect.y1);
            float inter_x2 = std::min(bboxes[good_idx].rect.x2, bboxes[tmp_i].rect.x2);
            float inter_y2 = std::min(bboxes[good_idx].rect.y2, bboxes[tmp_i].rect.y2);

            float w = std::max((inter_x2 - inter_x1 + 1), 0.0F);
            float h = std::max((inter_y2 - inter_y1 + 1), 0.0F);

            float inter_area = w * h;
            float area_1 = (bboxes[good_idx].rect.x2 - bboxes[good_idx].rect.x1 + 1) * (bboxes[good_idx].rect.y2 - bboxes[good_idx].rect.y1 + 1);
            float area_2 = (bboxes[tmp_i].rect.x2 - bboxes[tmp_i].rect.x1 + 1) * (bboxes[tmp_i].rect.y2 - bboxes[tmp_i].rect.y1 + 1);
            float o = inter_area / (area_1 + area_2 - inter_area);
            if (o <= threshold)
                idx.push_back(tmp_i);
        }
    }
    return bboxes_nms;
}


void RetinaFace::pre_process(Mat img, int inputW, int inputH, float &scale, Mat &resize){
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

void RetinaFace::detect(Mat &img, float threshold, float &scale, vector<FaceDetectInfo> &faceInfo)
{
    if (img.empty())
    {
        return;
    }
    int inputW = detectTrtNet->getNetWidth();
    int inputH = detectTrtNet->getNetHeight();

    cv::Mat resize;
    pre_process(img.clone(), inputW, inputH, scale, resize);
    resize.convertTo(resize, CV_32FC3);
    cvtColor(resize, resize, CV_BGR2RGB);

    vector<Mat> input_channels;
    float *input_data = cpuBuffers;

    for (int i = 0; i < detectTrtNet->getChannel(); ++i)
    {
        Mat channel(inputH, inputW, CV_32FC1, input_data);
        input_channels.push_back(channel);
        input_data += inputW * inputH;
    }

    split(resize, input_channels);
    float *inputData = (float *)detectTrtNet->getBuffer(0);
    cudaMemcpy(inputData, cpuBuffers, inputW * inputH * 3 * sizeof(float), cudaMemcpyHostToDevice);

    detectTrtNet->doInference(1);
    string name_bbox = "face_rpn_bbox_pred_";
    string name_score = "face_rpn_cls_prob_reshape_";
    string name_landmark = "face_rpn_landmark_pred_";

    for (size_t i = 0; i < _feat_stride_fpn.size(); i++)
    {
        string key = "stride" + std::to_string(_feat_stride_fpn[i]);

        string str = name_score + key;
        TrtBlob *score_blob = detectTrtNet->blob_by_name(str);
        std::vector<float> score = score_blob->result[0];
        std::vector<float>::iterator begin = score.begin() + score.size() / 2;
        std::vector<float>::iterator end = score.end();
        score = std::vector<float>(begin, end);

        str = name_bbox + key;
        TrtBlob *bbox_blob = detectTrtNet->blob_by_name(str);
        std::vector<float> bbox_delta = bbox_blob->result[0];

        str = name_landmark + key;
        TrtBlob *landmark_blob = detectTrtNet->blob_by_name(str);
        std::vector<float> landmark_delta = landmark_blob->result[0];

        int width = score_blob->outputDims.w();
        int height = score_blob->outputDims.h();
        size_t count = width * height;
        size_t num_anchor = _num_anchors[key];

        for (size_t num = 0; num < num_anchor; num++)
        {
            for (size_t j = 0; j < count; j++)
            {
                //置信度小于阈值跳过
                float conf = score[j + count * num];
                if (conf <= threshold)
                {
                    continue;
                }

                cv::Vec4f regress;
                float dx = bbox_delta[j + count * (0 + num * 4)];
                float dy = bbox_delta[j + count * (1 + num * 4)];
                float dw = bbox_delta[j + count * (2 + num * 4)];
                float dh = bbox_delta[j + count * (3 + num * 4)];
                regress = cv::Vec4f(dx, dy, dw, dh);

                //回归人脸框
                anchor_box rect = bbox_pred(_anchors[key][j + count * num], regress);
                //越界处理
                clip_boxes(rect, inputW, inputH);

                FacePts pts;
                for (size_t k = 0; k < 5; k++)
                {
                    pts.x[k] = landmark_delta[j + count * (num * 10 + k * 2)];
                    pts.y[k] = landmark_delta[j + count * (num * 10 + k * 2 + 1)];
                }
                //回归人脸关键点
                FacePts landmarks = landmark_pred(_anchors[key][j + count * num], pts);

                FaceDetectInfo tmp;
                tmp.score = conf;
                tmp.rect = rect;
                tmp.pts = landmarks;
                faceInfo.push_back(tmp);
            }
        }
    }
    //排序nms
    faceInfo = nms(faceInfo, nms_threshold);
}

void RetinaFace::detectBatchImages(vector<cv::Mat> imgs, float threshold,vector<float> &scales, vector<vector<FaceDetectInfo>> &faceInfos)
{
    //预处理
    int inputW = detectTrtNet->getNetWidth();
    int inputH = detectTrtNet->getNetHeight();

    // scales(imgs.size(), 1.0);

    double t2 = (double)getTickCount();

    for (size_t i = 0; i < imgs.size(); i++)
    {
        float sw = 1.0 * imgs[i].cols / inputW;
        float sh = 1.0 * imgs[i].rows / inputH;
        scales[i] = sw > sh ? sw : sh;
        scales[i] = scales[i] > 1.0 ? scales[i] : 1.0;

        if (sw > 1.0 || sh > 1.0)
        {
            if (sw > sh)
            {
                cv::resize(imgs[i], imgs[i], cv::Size(), 1 / sw, 1 / sw);
                cv::copyMakeBorder(imgs[i], imgs[i], 0, inputH - imgs[i].rows, 0, 0, cv::BORDER_CONSTANT, cv::Scalar(0));
            }
            else
            {
                cv::resize(imgs[i], imgs[i], cv::Size(), 1 / sh, 1 / sh);
                cv::copyMakeBorder(imgs[i], imgs[i], 0, 0, 0, inputW - imgs[i].cols, cv::BORDER_CONSTANT, cv::Scalar(0));
            }
        }
        else
        {
            //直接补边到目标大小
            cv::copyMakeBorder(imgs[i], imgs[i], 0, inputH - imgs[i].rows, 0, inputW - imgs[i].cols, cv::BORDER_CONSTANT, cv::Scalar(0));
        }

        //to float
        imgs[i].convertTo(imgs[i], CV_32FC3);

        //rgb
        cvtColor(imgs[i], imgs[i], CV_BGR2RGB);
    }

    //填充数据
    vector<vector<Mat>> input_channels;
    float *input_data = (float *)cpuBuffers;
    for (size_t j = 0; j < imgs.size(); j++)
    {
        vector<Mat> input_chans;
        for (int i = 0; i < detectTrtNet->getChannel(); ++i)
        {
            Mat channel(inputH, inputW, CV_32FC1, input_data);
            input_chans.push_back(channel);
            input_data += inputW * inputH;
        }
        input_channels.push_back(input_chans);
    }

    for (size_t j = 0; j < imgs.size(); j++)
    {
        split(imgs[j], input_channels[j]);
    }

    float *inputData = (float *)detectTrtNet->getBuffer(0);
    cudaMemcpy(inputData, cpuBuffers, imgs.size() * inputW * inputH * 3 * sizeof(float), cudaMemcpyHostToDevice);

    t2 = (double)getTickCount() - t2;
    double t1 = (double)getTickCount();
    detectTrtNet->doInference(imgs.size());
    t1 = (double)getTickCount() - t1;

    double post = (double)getTickCount();
    string name_bbox = "face_rpn_bbox_pred_";
    string name_score = "face_rpn_cls_prob_reshape_";
    string name_landmark = "face_rpn_landmark_pred_";

    
    for (size_t batch = 0; batch < imgs.size(); batch++)
    {
        vector<FaceDetectInfo> faceInfo;
        for (size_t i = 0; i < _feat_stride_fpn.size(); i++)
        {
            string key = "stride" + std::to_string(_feat_stride_fpn[i]);
            string str = name_score + key;
            TrtBlob *score_blob = detectTrtNet->blob_by_name(str);
            std::vector<float> score = score_blob->result[batch];
            std::vector<float>::iterator begin = score.begin() + score.size() / 2;
            std::vector<float>::iterator end = score.end();
            score = std::vector<float>(begin, end);

            str = name_bbox + key;
            TrtBlob *bbox_blob = detectTrtNet->blob_by_name(str);
            std::vector<float> bbox_delta = bbox_blob->result[batch];

            str = name_landmark + key;
            TrtBlob *landmark_blob = detectTrtNet->blob_by_name(str);
            std::vector<float> landmark_delta = landmark_blob->result[batch];

            int width = score_blob->outputDims.w();
            int height = score_blob->outputDims.h();
            size_t count = width * height;
            size_t num_anchor = _num_anchors[key];

            for (size_t num = 0; num < num_anchor; num++)
            {
                for (size_t j = 0; j < count; j++)
                {
                    //置信度小于阈值跳过
                    float conf = score[j + count * num];
                    if (conf <= threshold)
                    {
                        continue;
                    }

                    cv::Vec4f regress;
                    float dx = bbox_delta[j + count * (0 + num * 4)];
                    float dy = bbox_delta[j + count * (1 + num * 4)];
                    float dw = bbox_delta[j + count * (2 + num * 4)];
                    float dh = bbox_delta[j + count * (3 + num * 4)];
                    regress = cv::Vec4f(dx, dy, dw, dh);

                    //回归人脸框
                    anchor_box rect = bbox_pred(_anchors[key][j + count * num], regress);
                    //越界处理
                    clip_boxes(rect, inputW, inputH);

                    FacePts pts;
                    for (size_t k = 0; k < 5; k++)
                    {
                        pts.x[k] = landmark_delta[j + count * (num * 10 + k * 2)];
                        pts.y[k] = landmark_delta[j + count * (num * 10 + k * 2 + 1)];
                    }
                    //回归人脸关键点
                    FacePts landmarks = landmark_pred(_anchors[key][j + count * num], pts);

                    FaceDetectInfo tmp;
                    tmp.score = conf;
                    tmp.rect = rect;
                    tmp.pts = landmarks;
                    faceInfo.push_back(tmp);
                }
            }
        }

        faceInfos.push_back(faceInfo);
    }
    //排序nms
    for (size_t batch = 0; batch < imgs.size(); batch++)
    {
        faceInfos[batch] = nms(faceInfos[batch], nms_threshold);
    }
}

void RetinaFace::InitFaceDetector(string modelRootPath, string network_, float nms) {

    this->network = network_;
    this->nms_threshold = nms;

    //主干网络选择
    int fmc = 3;

    if (network == "ssh" || network == "vgg") {
        pixel_means[0] = 103.939;
        pixel_means[1] = 116.779;
        pixel_means[2] = 123.68;
    } else if (network == "net3") {
        _ratio = {1.0};
    } else if (network == "net3a") {
        _ratio = {1.0, 1.5};
    } else if (network == "net6") { //like pyramidbox or s3fd
        fmc = 6;
    } else if (network == "net5") { //retinaface
        fmc = 5;
    } else if (network == "net5a") {
        fmc = 5;
        _ratio = {1.0, 1.5};
    } else if (network == "net4") {
        fmc = 4;
    } else if (network == "net5a") {
        fmc = 4;
        _ratio = {1.0, 1.5};
    } else {
        std::cout << "network setting error" << network << std::endl;
    }

    //anchor配置
    if (fmc == 3) {
        _feat_stride_fpn = {32, 16, 8};
        anchor_cfg tmp;
        tmp.SCALES = {32, 16};
        tmp.BASE_SIZE = 16;
        tmp.RATIOS = _ratio;
        tmp.ALLOWED_BORDER = 9999;
        tmp.STRIDE = 32;
        cfg.push_back(tmp);

        tmp.SCALES = {8, 4};
        tmp.BASE_SIZE = 16;
        tmp.RATIOS = _ratio;
        tmp.ALLOWED_BORDER = 9999;
        tmp.STRIDE = 16;
        cfg.push_back(tmp);

        tmp.SCALES = {2, 1};
        tmp.BASE_SIZE = 16;
        tmp.RATIOS = _ratio;
        tmp.ALLOWED_BORDER = 9999;
        tmp.STRIDE = 8;
        cfg.push_back(tmp);
    } else {
        std::cout << "please reconfig anchor_cfg" << network << std::endl;
    }

    detectTrtNet = new TrtRetinaFaceNet("retina");
    detectTrtNet->buildTrtContext(modelRootPath + "/mnet.prototxt", modelRootPath + "/mnet.caffemodel",
                                  modelRootPath + "/mnet.engine");

    int maxbatchsize = detectTrtNet->getMaxBatchSize();
    int channels = detectTrtNet->getChannel();
    int inputW = detectTrtNet->getNetWidth();
    int inputH = detectTrtNet->getNetHeight();
    //
    int inputsize = maxbatchsize * channels * inputW * inputH * sizeof(float);
    cpuBuffers = (float *)malloc(inputsize);
    memset(cpuBuffers, 0, inputsize);

    vector<int> outputW = detectTrtNet->getOutputWidth();
    vector<int> outputH = detectTrtNet->getOutputHeight();

    bool dense_anchor = false;
    vector<vector<anchor_box>> anchors_fpn = generate_anchors_fpn(dense_anchor, cfg);
    for (size_t i = 0; i < anchors_fpn.size(); i++)
    {
        int stride = _feat_stride_fpn[i];
        string key = "stride" + std::to_string(_feat_stride_fpn[i]);
        _anchors_fpn[key] = anchors_fpn[i];
        _num_anchors[key] = anchors_fpn[i].size();
        //有三组不同输出宽高
        _anchors[key] = anchors_plane(outputH[i], outputW[i], stride, _anchors_fpn[key]);
    }
    isInitDetector = true;
}

void RetinaFace::InitFaceClassifier(string modelRootPath, string network) {

    if(!isInitClassifier || goggleMaskClassify == nullptr) {
        goggleMaskClassify = new GoggleMaskClassify(modelRootPath, network);
        isInitClassifier = true;
    }

}

void RetinaFace::detectFaces(const std::string imgPath, float threshold, float &scale, vector<FaceDetectInfo> &faceInfo) {
    if(isInitDetector) {
        cv::Mat img = cv::imread(imgPath);
        detect(img, threshold, scale, faceInfo);
    }
}

void RetinaFace::detectFaces(cv::Mat img, float threshold, float &scale, vector<FaceDetectInfo> &faceInfo) {
    if(isInitDetector) {
        detect(img, threshold, scale, faceInfo);
    }
}

void RetinaFace::classifyFace(const std::string faceImgPath, std::vector<float> &outputs) {
    if(isInitClassifier && goggleMaskClassify != nullptr) {
        cv::Mat face = cv::imread(faceImgPath);
        float scale = 1;
        goggleMaskClassify->classify(face, outputs,scale);
    }
}

void RetinaFace::classifyFace(cv::Mat faceImg, std::vector<float> &outputs) {
    if(isInitClassifier && goggleMaskClassify != nullptr) {
        float scale = 1;
        goggleMaskClassify->classify(faceImg, outputs,scale);
    }
}

GoggleMaskApi * GoggleMaskApi::CreateGMApi() {
    RetinaFace *retinaFace = new RetinaFace();
    return (GoggleMaskApi*)retinaFace;
}

void GoggleMaskApi::DestroyGMApi(GoggleMaskApi **api) {
    if (*api){
        delete *api;
        *api = nullptr;
    }

}

