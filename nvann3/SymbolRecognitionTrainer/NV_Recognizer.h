#pragma once
#include "MomentsRecognizer.h"
class NV_Recognizer :
	public MomentsRecognizer
{
public:
	NV_Recognizer();
	~NV_Recognizer();
	cv::Mat MomentsToInput(fe::ComplexMoments& moments);
	std::string  OutputToValue(cv::Mat output);
	bool Train(
		std::map<std::string, std::vector<fe::ComplexMoments>> moments,
		std::vector<int> layers,
		int max_iters = 100000,
		float eps = 0.1,
		float speed = 0.1);
};

