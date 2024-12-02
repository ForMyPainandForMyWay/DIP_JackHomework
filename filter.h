#pragma once
#include <opencv2/opencv.hpp>
#include <thread>
#include <future>

using namespace cv;
using namespace std;


// �����˲�
template<class Filter>
Mat filter_process(const Mat& target, const Filter& filter);

template<class Filter>
Mat solo_process(const Mat& target, const Filter& filter);

// �����˲�
template<class Filter>
Mat filter_process_parallel(const Mat& target, const Filter& filter);

template<class Filter>
void solo_process_parallel(const Mat& target, const Filter& filter, promise<Mat> promiseObj);

// ���о��
float colv(const Mat& orign, int midrow, int midcolumn, const Mat& kernel, const bool ProcessBrink);


// �˲�����(����),��������ͨ��ͼ�������ӽ����˲�����,���غϲ����ͼ��
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


// ���Ӵ�����(����)��ʹ�ô�������ӷ��ʹ��� ��ͨ��Mat ��ÿһ������
template<class Filter>
Mat solo_process(const Mat& target, const Filter& filter)
{
	Mat img = Mat::zeros(target.rows, target.cols, CV_8UC1);
	int covsize = (filter.rows - 1) / 2;

	// ����ԭͼ
	for (int row = 0; row < target.rows; row++) {
		for (int column = 0; column < target.cols; column++) {
			
			//if (row - covsize < 0 or column - covsize < 0) { 
				//img.at<uchar>(row, column) = target.at<uchar>(row, column); 
			//}
			// ��Ԫ�ش���
			//else img.at<uchar>(row, column) = filter.process(target, row, column);
			img.at<uchar>(row, column) = filter.process(target, row, column);
		}
	}
	return img;
}

// �˲�����(����),��������ͨ��ͼ�������ӽ����˲�����,���غϲ����ͼ��
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

// ���Ӵ�����(�벢��)��ʹ�ô�������ӷ��ʹ��� ��ͨ��Mat ��ÿһ������
template<class Filter>
void solo_process_parallel(const Mat& target, const Filter& filter, promise<Mat> promiseObj) {
	Mat img = Mat::zeros(target.rows, target.cols, CV_8UC1);
	int covsize = (filter.rows - 1) / 2;


	// ����ԭͼ
	for (int row = 0; row < target.rows; row++) {
		for (int column = 0; column < target.cols; column++) {
			// ��Ԫ�ؾ��
			img.at<uchar>(row, column) = filter.process(target, row, column);
		}
	}
	promiseObj.set_value(img);
}

// ���ӻ���,ֻ������
class MyFilter {
public:
	Mat kernel;
	int rows;
	int cols;
	// ���Ӵ�����
	virtual uchar process(const Mat& target, int row, int column) const;

	// ������ǿ����
	Mat enhance_parallel(Mat img) const;
	Mat enhance(Mat img) const;

	MyFilter();
	MyFilter(Mat kernel);
};

// Laplace����
class MyLaplace : public MyFilter {
public:
	// ���ع��캯��,Ĭ��Ϊ������ǿ������
	MyLaplace(Mat kernel = (cv::Mat_<float>(3, 3) << 0, 1, 0, 1, -4, 1, 0, 1, 0));
	MyLaplace(int mode, bool IsDark);

	// ͼ����ǿ,Ĭ��Ϊ������ǿ��
	Mat enhance_parallel(Mat im) const;
	Mat enhance(Mat img, bool IsDark = true) const;

protected:
	bool IsDark;
};

// sobel����
class MySobel : public MyFilter {
public:
	static Mat kernel;  // ����
	static Mat kernel_y;  // ����
	static int covsize;
	int T = 100;  // ��ֵ
	int method = 0;  // �˲�ģʽ

	// ���ع��캯��
	MySobel(int method = 0, int T = 100);

	// ��д���Ӵ���
	uchar process(const Mat& target, int row, int column) const;

	// ��ǿͼ��(����)
	Mat enhance_parallel(Mat img) const;
	// ��ǿͼ��(����)
	Mat enhance(Mat img) const;

	// ��������״̬
	void change(int method = 0, int T = 100);

private:
	// ֱ�ӽ�����ӳ��Ϊ���,��һ�������ʽ
	static uchar out_a(const uchar* temp, ...);

	// ��δ������ֵ��ֵ����һ���Ĺ�����м���(x+y),�ڶ��������ʽ
	static uchar out_b(const uchar* temp, ...);

	// ��������ֵ��ֵӳ�䵽�̶��Ҷȼ�,δ�����İ���һ���Ĺ�����м���,�����������ʽ
	static uchar out_c(const uchar* temp, ...);

	// ��������ֵ��ֱֵ�������δ������ӳ�䵽�Ҷȼ�,�����������ʽ
	static uchar out_d(const uchar* temp, ...);

	// ��������ֵ��ֵӳ�䵽�Ҷȼ���δ������ӳ�䵽��һ�Ҷȼ�,�����������ʽ
	static uchar out_e(const uchar* temp, ...);
};

// mean����
class MyMean : public MyFilter {
public:

	MyMean(Mat kernel = (cv::Mat_<float>(3, 3) << 1, 1, 1, 1, 1, 1, 1, 1, 1));

	// ��д������
	uchar process(const Mat& target, int row, int column) const;
};