#pragma once
#include "ANN.h"
#include <iostream>

class NeuralRealisation : public ANN::ANeuralNetwork
{
public:
	ANNDLL_API NeuralRealisation()
	{
	}
	ANNDLL_API NeuralRealisation(std::vector<size_t> conf, ANeuralNetwork::ActivationType aType, float scale);

	ANNDLL_API std::string GetType();
	ANNDLL_API std::vector<float> Predict(std::vector<float> & input);
	
	ANNDLL_API ~NeuralRealisation();
protected:
	ANNDLL_API void SetConf(std::vector<size_t> conf)
	{
		this->configuration = conf;
	}
	ANNDLL_API void SetAcT(ANeuralNetwork::ActivationType aType)
	{
		this->activation_type = aType;
	}
	ANNDLL_API void SetSc(float scale)
	{
		this->scale = scale;
	}
};

