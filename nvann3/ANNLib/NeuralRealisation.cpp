#define ANNDLL_EXPORTS
#include "NeuralRealisation.h"
using namespace std;

NeuralRealisation::NeuralRealisation(vector<size_t> conf, ANeuralNetwork::ActivationType aType, float scale)
{
	this->configuration = conf;
	this->activation_type = aType;
	this->scale = scale;

	if (conf.empty())
		return;
}

NeuralRealisation::~NeuralRealisation()
{
}

string NeuralRealisation::GetType()
{
	return ("This is SkyNet which will consume humankind!");
}

vector<float> NeuralRealisation::Predict(vector<float> & input)
{
	/*
	if (!is_trained || configuration.empty() || configuration[0] != input.size())
	{
		cout << "Problems!" << endl;
	}
	*/
	vector<float> prev_out = input;
	vector<float> cur_out;

	for (size_t layer_idx = 0; layer_idx < configuration.size() - 1; layer_idx++)
	{
		cur_out.resize(configuration[layer_idx + 1], 0);

		for (size_t to_idx = 0; to_idx < configuration[layer_idx + 1]; to_idx++)
		{
			for (size_t from_idx = 0; from_idx < configuration[layer_idx]; from_idx++)
			{
				cur_out[to_idx] += weights[layer_idx][from_idx][to_idx] * prev_out[from_idx];
			}
			cur_out[to_idx] = Activation(cur_out[to_idx]);
		}
		prev_out = cur_out;
	}
	
	return prev_out;
}

ANNDLL_API shared_ptr<ANN::ANeuralNetwork> ANN::CreateNeuralNetwork(
	vector<size_t> & config,
	ANN::ANeuralNetwork::ActivationType actType,
	float scale)
{
	return make_shared<NeuralRealisation>(config, actType, scale);
}

