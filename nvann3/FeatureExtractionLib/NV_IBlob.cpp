#include "NV_IBlob.h"



NV_IBlob::NV_IBlob()
{
}


NV_IBlob::~NV_IBlob()
{
}

std::string NV_IBlob::GetType()
{
	return "NV Iblob chek";
}

std::vector<cv::Mat> NV_IBlob::DetectBlobs(cv::Mat image)
{
	std::vector <cv::Mat> res;
	cv::Mat binary(image.rows, image.cols, CV_8UC1);
	threshold(image, binary, 127, 255, cv::THRESH_BINARY);
	//imshow("binary", binary);
	//waitKey(0);
	std::vector<std::vector<cv::Point>> contours;
	std::vector<cv::Vec4i> hierarchy;
	findContours(binary, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);
	cv::Mat drawing = cv::Mat::zeros(binary.size(), CV_8UC3);
	cv::RNG rng(0xFFFFFFFF);
	for (int idx = 0; idx >= 0; idx = hierarchy[idx][0])
	{
		cv::Scalar colour = cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
		drawContours(drawing, contours, idx, colour, CV_FILLED, cv::LineTypes::LINE_8, hierarchy);
	}

	for (int idx = 0; idx >= 0; idx = hierarchy[idx][0])
	{
		cv::Point2f center;
		float radius;
		minEnclosingCircle(contours[idx], center, radius);
		res.push_back(cv::Mat::zeros(round((double)radius) * 2, round((double)radius) * 2, CV_8UC1));
		cv::Scalar white = cv::Scalar(255, 255, 255);
		cv::Point2f offset = -center + cv::Point2f(radius, radius);
		drawContours(res.back(), contours, idx, white, CV_FILLED, cv::LineTypes::LINE_8, hierarchy, 255, offset);
		//	imshow("Symbol", res.back());
		//	waitKey(0);
	}
	return res;
}

std::vector<cv::Mat> NV_IBlob::NormalizeBlobs(std::vector<cv::Mat>& blobs, int side)
{
	std::vector<cv::Mat> normalized;
	for (auto& blob : blobs)
	{
		cv::Mat m;
		resize(blob, m, cv::Size(side, side));
		normalized.push_back(m);
	}
	return normalized;
}
