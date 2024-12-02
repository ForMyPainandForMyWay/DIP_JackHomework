#include <opencv2/opencv.hpp>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <stdarg.h>
#include <thread>
#include <future>

#include "filter.h"  // 核心算法的实现

// 题目要求的函数
Mat myconvolution(Mat inputImage, Mat mytemplate) {
	Mat result;
	
	cv::Mat result1;
	cv::Mat result2;
	cv::Mat result3_x;
	cv::Mat result3_y;

	Mat mat1 = (cv::Mat_<float>(3, 3) << 1, 1, 1, 1, 1, 1, 1, 1, 1);  // 邻域
	Mat mat2 = (cv::Mat_<float>(3, 3) << 0, -1, 0, -1, 4, -1, 0, -1, 0);  // 拉普拉斯
	Mat mat3_x = (cv::Mat_<float>(3, 3) << -1, 0, 1, -2, 0, 2, -1, 0, 1);  // sobel_x
	Mat mat3_y = (cv::Mat_<float>(3, 3) << -1, -2, -1, 0, 0, 0, 1, 2, 1);  // sobel_y

	// 检查是否所有像素都相同
	cv::compare(mytemplate, mat1, result1, cv::CMP_EQ);
	cv::compare(mytemplate, mat2, result2, cv::CMP_EQ);
	cv::compare(mytemplate, mat3_x, result3_x, cv::CMP_EQ);
	cv::compare(mytemplate, mat3_y, result3_y, cv::CMP_EQ);

	bool notMean = cv::countNonZero(result1) == 0;
	bool notL = cv::countNonZero(result2) == 0;
	bool notS_x = cv::countNonZero(result3_x) == 0;
	bool notS_y = cv::countNonZero(result3_y) == 0;	

	// 均值平滑化
	if (countNonZero(result1)==9) {
		MyMean mfilter;
		result = mfilter.enhance_parallel(inputImage);
	}
	// 拉普拉斯4邻域
	else if (countNonZero(result2)==9) {
		MyLaplace lfilter(4, true);
		result = lfilter.enhance_parallel(inputImage);
		//result = filter_process_parallel(inputImage, lfilter);
	}
	// sobel处理,只获取梯度图像
	else if (countNonZero(result3_x)==9 or countNonZero(result3_y) == 9) {

		MySobel sfilter;
		result = sfilter.enhance_parallel(inputImage);
	}
	// 其它卷积方式,如拉普拉斯增强核、高通滤波
	else {

		MyFilter filter(mytemplate);
		result = filter.enhance_parallel(inputImage);
	}
	return result;
}


int main() {

	Mat av_template = (Mat_<float>(3, 3) << 1, 1, 1, 1, 1, 1, 1, 1, 1);

	Mat la_template = (Mat_<float>(3, 3) << 0, -1, 0, -1, 5, -1, 0, -1, 0);

	Mat sobel_x = (Mat_<float>(3, 3) << -1, 0, 1, -2, 0, 2, -1, 0, 1);

	Mat sobel_y = (Mat_<float>(3, 3) << -1, -2, -1, 0, 0, 0, 1, 2, 1);

	string path = R"(C:\Users\yyd\Pictures\Screenshots\new.png)";

	Mat img = imread(path, 1);


	if (img.empty()) {
		cout << "图像为空" << endl;
		return 0;
	}
	imshow("org", img);

	
	Mat av_outImage, la_outImage, sobel_outImage, display;
	
	av_outImage = myconvolution(img, av_template);
	imshow("av", av_outImage);
	
	la_outImage = myconvolution(img, la_template);
	imshow("la", la_outImage);
	
	
	sobel_outImage = myconvolution(img, sobel_x);
	sobel_outImage = myconvolution(img, sobel_y);
	imshow("sobel", sobel_outImage);
	

	waitKey(0);
	destroyAllWindows();
	
	return 0;
}