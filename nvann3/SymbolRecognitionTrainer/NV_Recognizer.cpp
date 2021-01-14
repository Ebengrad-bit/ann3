#include "NV_Recognizer.h"

#include <sstream>


NV_Recognizer::NV_Recognizer()
{
	pAnn = cv::ml::ANN_MLP::create();
}


NV_Recognizer::~NV_Recognizer()
{
}

cv::Mat  NV_Recognizer::MomentsToInput(fe::ComplexMoments& moments)
{
	cv::Mat res = cv::Mat::zeros(1, moments.im.rows, CV_32F);
	//std::vector<float> buf;
	for (int i = 0; i < moments.im.rows; i++)
	{
		float im = moments.im.at<double>(i);
		float re = moments.re.at<double>(i);
		//buf.push_back(sqrt(im*im+re*re));
		//res.at<float>(i) = sqrt(im*im + re*re);
		//res.at<float>(i * 2) = im;
		//res.at<float>(i * 2 + 1) = re;
		res.at<float>(i) = sqrt(im*im + re*re);
	}
	return res;
}


std::string  NV_Recognizer::OutputToValue(cv::Mat output)
{
	std::string result;
	std::vector<double> buf;
	buf.resize(9);
	for (int i = 0; i < 9; i++)
		buf[i] = output.at<float>(i);
	double dd = 0;
	int k = 0;
	for (int i = 0; i < buf.size(); i++)
	{
		if (dd < buf[i])
		{
			k = i;
			dd = buf[i];
		}
	}
	std::stringstream ss("");
	ss << k;
	result += ss.str();
	return result;
}


bool NV_Recognizer::Train(
	std::map<std::string, std::vector<fe::ComplexMoments>> moments,
	std::vector<int> layers,
	int max_iters,
	float eps,
	float speed)
{
	cv::Mat lay = cv::Mat(1, layers.size() + 2, CV_16U);
	lay.col(0) = cv::Scalar(moments.at("0").front().im.rows);
	for (int i = 0; i < layers.size(); i++)
		lay.col(i + 1) = cv::Scalar(layers[i]);
	lay.col(layers.size() + 1) = cv::Scalar(9);
	pAnn->setLayerSizes(lay);

	cv::TermCriteria termCrit(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, max_iters, eps);
	pAnn->setTermCriteria(termCrit);
	pAnn->setActivationFunction(cv::ml::ANN_MLP::ActivationFunctions::SIGMOID_SYM, 1.0, 1.0);
	pAnn->setTrainMethod(cv::ml::ANN_MLP::TrainingMethods::BACKPROP);
	pAnn->setBackpropMomentumScale(speed);
	pAnn->setBackpropWeightScale(0.1);

	int moments_size = 0;
	for (int n = 0; n < 9; n++) { moments_size += moments[std::to_string(n)].size(); }
	cv::Mat input = cv::Mat::zeros(moments.at("0").front().im.rows, moments_size, CV_32F);
	cv::Mat response = cv::Mat::zeros(9, moments_size, CV_32F);
	moments_size = 0;
	for (int n = 0; n < 9; n++)
	{
		std::vector<fe::ComplexMoments> numom = moments[std::to_string(n)];
		for (uint i = 0; i < numom.size(); i++)
		{
			cv::Mat datarow = MomentsToInput(numom[i]);
			for (int x = 0; x < moments.at("0").front().im.rows; x++)
			{
				input.at<float>(x, moments_size + i) = datarow.at<float>(x);
			}

			for (int x = 0; x < 9; x++)
			{
				response.at<float>(x, moments_size + i) = -1.0f;
			}
			response.at<float>(n, moments_size + i) = 1.0f;
		}
		moments_size += numom.size();
	}
	auto trainData = cv::ml::TrainData::create(input, cv::ml::COL_SAMPLE, response);
	bool done = pAnn->train(trainData, cv::ml::ANN_MLP::NO_INPUT_SCALE + cv::ml::ANN_MLP::NO_OUTPUT_SCALE);
	return pAnn->isTrained();
}