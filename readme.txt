1, commond :(in ROOT PATH)
	mkdir build
	cd build
	cmake ..
	make

2, 编译完成后会生成动态链接库: libgogglemask.so 和 可运行文件 GoggleMask，对应的main函数在GoggleMask文件夹下。

3, 接口头文件：ROOTPATH/goggleMask/goggleMaskAPI.h, 接口使用示例：ROOTPATH/interface
	
	把编译生成的libgoggleMask.so动态链接库复制到 ROOTPATH/interface/lib 目录下
	mkdir build & cd build & cmake .. & make & ./goggleMask

4, 头文件 goggleMaskAPI.h : (已经在文件中详细说明每个函数的功能)

	1) InitFaceDetector(string modelRootPath, string network = "net3", float nms = 0.4) 
		说明：初始化人脸检测模型
		输入：modelRootPath : 模型文件所在的文件夹路径，该文件夹下必须包含 mnet.prototxt 和 mnet.caffemodel 模型文件
		      network : 网络名称，使用“net3”
		      num : 非极大抑制阈值
	2) InitFaceClassifier(string modelRootPath, string network="goggleMask")
		说明：初始化人脸分类模型
		输入：modelRootPath : 模型文件所在的文件夹路径，该文件夹下必须包含 goggle_classify.prototxt 和 goggle_classify.caffemodel 模型文件
		     network : 网络名称，可以忽略
	3) void detectFaces(std::string imgPath, float threshold, float &scale, vector<FaceDetectInfo> &faceInfo)
	   void detectFaces(cv::Mat img, float threshold, float &scale, vector<FaceDetectInfo> &faceInfo)
		说明：检测人脸
		输入：imgPath : 待检测人脸图像路径 ; img : 待检测人脸图像数据矩阵
		      threshold : 检测阈值
		输出：scale : 图像缩放的尺度，需要将返回结果的边界框乘以该尺度大小
		      faceInfo : 检测的人脸数据结构体，包含人脸边界框、得分、和5个人脸特征点
	4) void classifyFace(std::string faceImgPath, std::vector<float> &outputs)
	   void classifyFace(cv::Mat faceImg, std::vector<float> &outputs)
		说明：人脸分类，是否戴护目镜口罩
		输入：faceImgPath : 待分类的人脸图像路径 ; faceImg : 待分类的人脸数据矩阵
		输出：outputs : 分类结果，大小为4，依次表示 戴护目镜和口罩的得分、戴护目镜不戴口罩的得分、戴口罩不戴护目镜的得分、不戴护目镜不戴口罩的得分
	5) void detectGoggle(std::string imgPath, std::vector<GoggleDetectInfo> &outputs, float faceThreshold = 0.5)
	   void detectGoggle(cv::Mat img, std::vector<GoggleDetectInfo> &outputs, float faceThreshold = 0.5)
		说明：图像检测护目镜口罩
		输入：imgPath : 待检测的图像路径; img : 待检测的图像数据矩阵
		      faceThreshold : 人脸检测的阈值，默认为0.5
		输出：outputs : GoggleDetectInfo结构体的数组，表示每个人脸的检测结果
	6) bool isInitFaceDetector()
		说明：是否初始化人脸检测模型
		输出：bool, 初始化或没有初始化
	7) bool isInitFaceClassifier()
		说明：是否初始化人脸分类模型
		输出：bool, 初始化或没有初始化
	
	结构体：
		struct GoggleDetectInfo {
			float score;           // 检测的得分
			int label;             // 0--goggleMask, 1--goggle, 2--mask, 3--no
			float x1, y1, x2, y2;  // 人脸边界框：左上角点和右下角点
		};
	
5, note
	当模型第一次运行时可能会报错，这个没有问题，是因为在caffe模型转化为TensorRT模型时不能同时转换两个模型，再次运行后即可成功运行。
	运行成功后会在模型文件夹下生成对应的.engine文件：mnet.engine 和 goggle_classify.engine
