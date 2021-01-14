#include "Individual.h"



ga::Individual::Individual(
	std::vector<size_t> & configuration,
	ANN::ANeuralNetwork::ActivationType activation_type,
	float scale,
	std::string data_source
	)
{
	this->configuration = configuration;
	this->activation_type = activation_type;
	this->scale = scale;
	this->RandomInit();
	if (ANN::LoadData(data_source, this->inputs, this->outputs) == false)
		{ throw "An error occured while loading training data!"; }
}


ga::Individual::~Individual()
{
}

std::shared_ptr<ga::IIndividual> ga::Individual::Mutation()
{
	float mag = 0.4f;
	Individual mutant(this->configuration, this->activation_type, this->scale, "xor.data");
	mutant.weights = this->weights;
	std::random_device rd; std::mt19937 randm(rd());
	for (int l = 0; l < mutant.weights.size(); l++)
	{
		for (int n = 0; n < mutant.weights[l].size(); n++)
		{
			for (int f = 0; f < mutant.weights[l][n].size(); f++)
			{
				mutant.weights[l][n][f] *= (1.0f + mag - 2.0f * mag * (float)(randm()) / (float)(randm.max()));
				/*if((float)randm()/randm.max()>0.5)
					mutant.weights[l][n][f] += 0.5*(float)randm() / (float)randm.max();
				else
					mutant.weights[l][n][f] -= 0.5*(float)randm() / (float)randm.max();*/
			}
		}
	}
	return std::make_shared<Individual>(mutant);
}

std::shared_ptr<ga::IIndividual> ga::Individual::Crossover(std::shared_ptr<IIndividual> individual)
{
	Individual* child = reinterpret_cast<Individual*>(individual.get());
	std::random_device rd; std::mt19937 randm(rd());
	for (int l = 0; l < child->weights.size(); l++)
	{
		for (int n = 0; n < child->weights[l].size(); n++)
		{
			for (int f = 0; f < child->weights[l][n].size(); f++)
			{
				if (randm() > randm.max() / 2)
				{ 
					child->weights[l][n][f] = this->weights[l][n][f];
				}
			}
		}
	}
	return std::make_shared<ga::Individual>(*child);
}

std::pair<int, int> ga::Individual::Spare(std::shared_ptr<IIndividual> individual)
{
	float eps1 = 0.0f, eps2 = 0.0f;
	for (int i = 0; i < inputs.size(); i++)
	{
		auto prediction1 = this->MakeDecision(inputs[i]);
		auto prediction2 = individual->MakeDecision(inputs[i]);
		for (int j = 0; j < outputs[i].size(); j++) 
		{ 
			eps1 += (outputs[i][j] - prediction1[j]) * (outputs[i][j] - prediction1[j])*10;
			eps2 += (outputs[i][j] - prediction2[j])*(outputs[i][j] - prediction2[j]) * 10;
		}
	}
	if (eps1 == eps2)
		return std::make_pair<int, int>(0, 0);
	return ((eps1 < eps2) ? std::make_pair<int, int>(1, -3) : std::make_pair<int, int>(-3, 1));
	/*auto res = std::make_pair<int, int>(0, 0);
	float seps1 = 0; float seps2 = 0;
	for (int i = 0; i < inputs.size(); i++)
	{
		auto prediction1 = this->MakeDecision(inputs[i]);
		auto prediction2 = individual->MakeDecision(inputs[i]);
		float eps1 = 0;
		float eps2 = 0;
		for (int j = 0; j < outputs[i].size(); j++)
		{
			eps1 += abs(outputs[i][j] - prediction1[j]);// * (outputs[i][j] - prediction1[j])*10;
			eps2 += abs(outputs[i][j] - prediction2[j]);// *(outputs[i][j] - prediction2[j]) * 10;
		}
		if (eps1 == eps2)
			continue;
		if (eps1 < eps2)
			res.first++;
		else
		{
			res.second--;
		}
		seps1 += eps1;
		seps2 += eps2;
	}
	if (seps1 < seps2)
	{
		res.first ++;
		res.second --;
	}
	else
	{
		res.second ++;
		res.first --;
	}
	return res;*/
}

std::vector<float> ga::Individual::MakeDecision(std::vector<float>& input)
{
	return this->Predict(input);
}

std::shared_ptr<ga::IIndividual> ga::Individual::Clone()
{
	Individual* clone = new Individual(this->configuration, this->activation_type, this->scale, "xor.data");
	clone->is_trained = this->is_trained;
	clone->weights = this->weights;
	return std::make_shared<ga::Individual>(*clone);
}

std::string ga::Individual::GetType()
{
	return "nv";
}

std::vector<float> ga::Individual::Predict(std::vector<float> & input)
{
	std::vector<float> result, buffer = input;
	for (size_t layer_idx = 0; layer_idx < (configuration.size() - 1); layer_idx++)
	{
		result.resize(configuration[layer_idx + 1]);
		for (size_t to_idx = 0; to_idx < configuration[layer_idx + 1]; to_idx++)
		{
			result[to_idx] = 0.0f;
			for (size_t from_idx = 0; from_idx < configuration[layer_idx]; from_idx++)
				result[to_idx] += buffer[from_idx] * weights[layer_idx][from_idx][to_idx];
			result[to_idx] = Activation(result[to_idx]);
		}
		buffer = result;
	}
	return result;
}