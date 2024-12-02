#include "filter.h"

// 卷积函数(串行),仅仅计算卷积的float结果，不保证数值大小
float colv(const Mat& orign, int midrow, int midcolumn, const Mat& kernel, const bool ProcessBrink = false) {
	float mid = 0;
	int covsize = (kernel.rows - 1) / 2;

	if (not ProcessBrink and (midrow - covsize < 0 or midcolumn - covsize < 0)) mid = orign.at<uchar>(midrow, midcolumn);
	else {
		for (int row = midrow - covsize; row <= midrow + covsize; row++) {
			if (row < orign.rows and row >= 0) {
				for (int column = midcolumn - covsize; column <= midcolumn + covsize; column++) {
					if (column < orign.cols and column >= 0) {
						mid += ((int)(orign.at<uchar>(row, column)) * kernel.at<float>(row - midrow + covsize, column - midcolumn + covsize));
					}
				}
			}
		}
	}
	return mid;
}

MyFilter::MyFilter(Mat kernel) {
	this->kernel = kernel;
	this->rows = kernel.rows;
	this->cols = kernel.cols;
}

// 提供uchar类型的数据
uchar MyFilter::process(const Mat& target, int row, int column) const {
	// 截断输出
	return saturate_cast<uchar>(colv(target, row, column, this->kernel));
}

Mat MyFilter::enhance_parallel(Mat img) const {

	return filter_process_parallel(img, *this);
}

Mat MyFilter::enhance(Mat img) const {
	return filter_process(img, *this);
}

MyFilter::MyFilter() {}

MyLaplace::MyLaplace(Mat kernel) {
	this->kernel = kernel;
	this->IsDark = true;
	this->rows = kernel.rows;
	this->cols = kernel.cols;
}

MyLaplace::MyLaplace(int mode, bool IsDark) {
	this->IsDark = IsDark;
	this->rows = 3;
	this->cols = 3;
	if (mode == 4) {
		if (IsDark) {
			this->kernel = (cv::Mat_<float>(3, 3) << 0, 1, 0, 1, -4, 1, 0, 1, 0);
		}
		else {
			this->kernel = (cv::Mat_<float>(3, 3) << 0, -1, 0, -1, 4, -1, 0, -1, 0);
		}
	}
	else {
		if (IsDark) {
			this->kernel = (cv::Mat_<float>(3, 3) << -1, -1, -1, -1, 8, -1, -1, -1, -1);
		}
		else {
			this->kernel = (cv::Mat_<float>(3, 3) << 1, 1, 1, 1, -8, 1, 1, 1, 1);
		}
	}
}


Mat MyLaplace::enhance_parallel(Mat img) const
{
	// 出于系统解耦性以及对比对失衡的考虑，弃用原先直接相加后归一化的操作
	/*
	img.convertTo(img, CV_32F);

	Mat dst; // 这将是归一化后的Mat
	temp.convertTo(dst, CV_32F); // 首先转换为浮点型，以避免溢出
	dst = img + dst;

	double minVal, maxVal;
	minMaxLoc(temp, &minVal, &maxVal); // 找到最小值和最大值
	double scale = 255.0 / (maxVal - minVal); // 计算缩放因子
	
	dst.convertTo(dst, CV_8U, scale, -minVal * scale); // 转换为无符号8位整型，并限制范围在0-255
	//imshow("ai", dst);
	*/

	if (this->IsDark)	return img - filter_process_parallel(img, *this);  // 暗线增强型
	else return img + filter_process_parallel(img, *this);
}

Mat MyLaplace::enhance(Mat img, bool IsDark) const {
	return img - filter_process(img, *this);
}

MySobel::MySobel(int method, int T) {
	this->method = method;
	if (not method and not T) { this->T = 100; }
	else { this->T = T; }
	this->rows = 3;
	this->cols = 3;
}

uchar MySobel::process(const Mat& target, int row, int column) const {
	float grad_x = saturate_cast<uchar>((abs(colv(target, row, column, kernel))));
	float grad_y = saturate_cast<uchar>((abs(colv(target, row, column, kernel_y))));
	//float temp = cv::sqrt(cv::pow(grad_x, 2) + cv::pow(grad_y, 2));
	uchar grad = saturate_cast<uchar>(0.5*grad_y+0.5*grad_x);


	// 共5种输出方式
	if (not this->method) { return out_a(&grad); }
	else if (this->method == 1) { return out_b(&grad, this->T, target.at<uchar>(row, column)); }
	else if (this->method == 2) { return out_c(&grad, this->T, target.at<uchar>(row, column)); }
	else if (this->method == 3) { return out_d(&grad, this->T); }
	else if (this->method == 4) { return out_e(&grad, this->T); }
	else return out_a(&grad);
}

Mat MySobel::enhance_parallel(Mat img) const {
	return filter_process_parallel(img, *this);
}

Mat MySobel::enhance(Mat img) const {
	return filter_process(img, *this);
}

void MySobel::change(int method, int T) {
	this->method = method;
	if (not method and not T) { this->T = 100; }
	else { this->T = T; }
}

uchar MySobel::out_a(const uchar* temp, ...) {
	return *temp;
}

uchar MySobel::out_b(const uchar* temp, ...) {
	va_list args;
	va_start(args, temp);
	int T = va_arg(args, int);
	uchar dire = va_arg(args, uchar);
	va_end(args);

	if (*temp >= T) return *temp;
	return dire;
}

uchar MySobel::out_c(const uchar* temp, ...) {
	va_list args;
	va_start(args, temp);
	int T = va_arg(args, int);
	uchar dire = va_arg(args, uchar);
	va_end(args);

	if (*temp >= T) return 255;  // 灰度级白色
	return dire;
}

uchar MySobel::out_d(const uchar* temp, ...) {
	va_list args;
	va_start(args, temp);
	int T = va_arg(args, int);
	va_end(args);

	if (*temp >= T) return *temp;
	return 0;  // 灰度级黑色
}

uchar MySobel::out_e(const uchar* temp, ...) {
	va_list args;
	va_start(args, temp);
	int T = va_arg(args, int);
	va_end(args);

	if (*temp >= T) return 255;  // 灰度级白色
	return 0;  // 灰度级黑色
}

Mat MySobel::kernel = (cv::Mat_<float>(3, 3) << -1, 0, 1, -2, 0, 2, -1, 0, 1);
Mat MySobel::kernel_y = (cv::Mat_<float>(3, 3) << -1, -2, -1, 0, 0, 0, 1, 2, 1);
int MySobel::covsize = 3;

MyMean::MyMean(Mat kernel) {
	this->kernel = kernel;
	this->rows = kernel.rows;
	this->cols = kernel.cols;
}

uchar MyMean::process(const Mat& target, int row, int column) const {
	return (uchar)(int)(colv(target, row, column, this->kernel) / (this->cols * this->rows));
}