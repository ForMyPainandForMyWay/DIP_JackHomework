#pragma once
#include <opencv2/opencv.hpp>
#include <thread>
#include <future>

using namespace cv;
using namespace std;


// 串行滤波
template<class Filter>
Mat filter_process(const Mat& target, const Filter& filter);

template<class Filter>
Mat solo_process(const Mat& target, const Filter& filter);

// 并行滤波
template<class Filter>
Mat filter_process_parallel(const Mat& target, const Filter& filter);

template<class Filter>
void solo_process_parallel(const Mat& target, const Filter& filter, promise<Mat> promiseObj);

// 串行卷积
float colv(const Mat& orign, int midrow, int midcolumn, const Mat& kernel, const bool ProcessBrink);


// 滤波函数(串行),传入任意通道图像与算子进行滤波处理,返回合并后的图像
template<class Filter>
Mat filter_process(const Mat& target, const Filter& filter)
{
	vector<Mat> channels;
	Mat MergedImg;
	cv::split(target, channels);

	for (int i = 0; i < channels.size(); i++) {
		channels[i] = solo_process<Filter>(ref(channels[i]), ref(filter));
	}

	cv::merge(channels, MergedImg);
	return MergedImg;
}


// 算子处理函数(串行)，使用传入的算子泛型处理 单通道Mat 的每一个像素
template<class Filter>
Mat solo_process(const Mat& target, const Filter& filter)
{
	Mat img = Mat::zeros(target.rows, target.cols, CV_8UC1);
	int covsize = (filter.rows - 1) / 2;

	// 遍历原图
	for (int row = 0; row < target.rows; row++) {
		for (int column = 0; column < target.cols; column++) {
			
			//if (row - covsize < 0 or column - covsize < 0) { 
				//img.at<uchar>(row, column) = target.at<uchar>(row, column); 
			//}
			// 逐元素处理
			//else img.at<uchar>(row, column) = filter.process(target, row, column);
			img.at<uchar>(row, column) = filter.process(target, row, column);
		}
	}
	return img;
}

// 滤波函数(并行),传入任意通道图像与算子进行滤波处理,返回合并后的图像
template<class Filter>
Mat filter_process_parallel(const Mat& target, const Filter& filter) {
	vector<Mat> channels;
	Mat MergedImg;
	cv::split(target, channels);
	vector<thread> pool;
	vector<promise<Mat>> promise_pool;
	vector<future<Mat>> future_pool;

	for (int i = 0; i < channels.size(); i++) {
		promise_pool.emplace_back();
		future_pool.emplace_back(promise_pool[i].get_future());

		pool.emplace_back(thread(solo_process_parallel<Filter>, ref(channels[i]), ref(filter), move(promise_pool[i])));
	}

	for (int i = 0; i < channels.size(); i++) {
		channels[i] = future_pool[i].get();
		pool[i].join();
	}

	cv::merge(channels, MergedImg);
	return MergedImg;
}

// 算子处理函数(半并行)，使用传入的算子泛型处理 单通道Mat 的每一个像素
template<class Filter>
void solo_process_parallel(const Mat& target, const Filter& filter, promise<Mat> promiseObj) {
	Mat img = Mat::zeros(target.rows, target.cols, CV_8UC1);
	int covsize = (filter.rows - 1) / 2;


	// 遍历原图
	for (int row = 0; row < target.rows; row++) {
		for (int column = 0; column < target.cols; column++) {
			// 逐元素卷积
			img.at<uchar>(row, column) = filter.process(target, row, column);
		}
	}
	promiseObj.set_value(img);
}

// 算子基类,只计算卷积
class MyFilter {
public:
	Mat kernel;
	int rows;
	int cols;
	// 算子处理函数
	virtual uchar process(const Mat& target, int row, int column) const;

	// 算子增强函数
	Mat enhance_parallel(Mat img) const;
	Mat enhance(Mat img) const;

	MyFilter();
	MyFilter(Mat kernel);
};

// Laplace算子
class MyLaplace : public MyFilter {
public:
	// 重载构造函数,默认为暗线增强型算子
	MyLaplace(Mat kernel = (cv::Mat_<float>(3, 3) << 0, 1, 0, 1, -4, 1, 0, 1, 0));
	MyLaplace(int mode, bool IsDark);

	// 图像增强,默认为暗线增强型
	Mat enhance_parallel(Mat im) const;
	Mat enhance(Mat img, bool IsDark = true) const;

protected:
	bool IsDark;
};

// sobel算子
class MySobel : public MyFilter {
public:
	static Mat kernel;  // 横向
	static Mat kernel_y;  // 纵向
	static int covsize;
	int T = 100;  // 阈值
	int method = 0;  // 滤波模式

	// 重载构造函数
	MySobel(int method = 0, int T = 100);

	// 重写算子处理
	uchar process(const Mat& target, int row, int column) const;

	// 增强图像(并行)
	Mat enhance_parallel(Mat img) const;
	// 增强图像(串行)
	Mat enhance(Mat img) const;

	// 重置算子状态
	void change(int method = 0, int T = 100);

private:
	// 直接将输入映射为输出,第一种输出形式
	static uchar out_a(const uchar* temp, ...);

	// 将未超过阈值的值按照一定的规则进行计算(x+y),第二种输出形式
	static uchar out_b(const uchar* temp, ...);

	// 将超过阈值的值映射到固定灰度级,未超过的按照一定的规则进行计算,第三种输出形式
	static uchar out_c(const uchar* temp, ...);

	// 将超过阈值的值直接输出，未超过的映射到灰度级,第四种输出形式
	static uchar out_d(const uchar* temp, ...);

	// 将超过阈值的值映射到灰度级，未超过的映射到另一灰度级,第五种输出形式
	static uchar out_e(const uchar* temp, ...);
};

// mean算子
class MyMean : public MyFilter {
public:

	MyMean(Mat kernel = (cv::Mat_<float>(3, 3) << 1, 1, 1, 1, 1, 1, 1, 1, 1));

	// 重写处理函数
	uchar process(const Mat& target, int row, int column) const;
};