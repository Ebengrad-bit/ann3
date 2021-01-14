#define ANNDLL_EXPORTS
#include <ANN.h>
#include <fstream>
#include <iomanip>
#include <iostream>


bool ANN::ANeuralNetwork::Load(std::string filepath)
{
	std::ifstream file(filepath);
	if (!file.is_open()) return false;
	int buffer;
	const int CHAR_BUF_LEN = 100;
	char char_buffer[CHAR_BUF_LEN];
	file.getline(char_buffer, CHAR_BUF_LEN);
	std::string string_buffer = std::string(char_buffer);
	memset(char_buffer, 0, CHAR_BUF_LEN);
	if (string_buffer != std::string("activation type:"))
		throw "incorrect file format";
	file >> buffer;
	activation_type = (ActivationType)buffer;
	file.getline(char_buffer, CHAR_BUF_LEN);
	file.getline(char_buffer, CHAR_BUF_LEN);
	string_buffer = std::string(char_buffer);
	memset(char_buffer, 0, CHAR_BUF_LEN);
	if (string_buffer != std::string("activation scale:"))
		throw "incorrect file format";
	file >> scale;
	file.getline(char_buffer, CHAR_BUF_LEN);
	file.getline(char_buffer, CHAR_BUF_LEN);
	string_buffer = std::string(char_buffer);
	memset(char_buffer, 0, CHAR_BUF_LEN);
	if (string_buffer != std::string("configuration:"))
		throw "incorrect file format";
	file >> buffer;
	configuration.resize(buffer);
	for (size_t i = 0; i < configuration.size(); i++) {
		file >> configuration[i];
	}
	file.getline(char_buffer, CHAR_BUF_LEN);
	file.getline(char_buffer, CHAR_BUF_LEN);
	string_buffer = std::string(char_buffer);
	memset(char_buffer, 0, CHAR_BUF_LEN);
	if (string_buffer != std::string("weights:"))
		throw "incorrect file format";
	weights.resize(configuration.size() - 1);
	for (size_t layer_idx = 0; layer_idx < weights.size(); layer_idx++) {
		weights[layer_idx].resize(configuration[layer_idx]);
		for (size_t from_idx = 0; from_idx < weights[layer_idx].size(); from_idx++) {
			weights[layer_idx][from_idx].resize(configuration[layer_idx + 1]);
			for (size_t to_idx = 0; to_idx < weights[layer_idx][from_idx].size(); to_idx++) {
				file >> weights[layer_idx][from_idx][to_idx];
			}
		}
	}
	file.close();
	is_trained = true;
	return true;
}

bool ANN::ANeuralNetwork::Save(std::string filepath)
{
	if (!is_trained) return false;
	std::ofstream file(filepath);
	if (!file.is_open()) return false;
	file << std::setprecision(9);
	file << "activation type:" << std::endl;
	file << (int)activation_type << std::endl;
	file << "activation scale:" << std::endl;
	file << scale << std::endl;
	file << "configuration:" << std::endl;
	file << configuration.size() << "\t";
	for each (int neuron_count in configuration) {
		file << neuron_count << "\t";
	}
	file << std::endl << "weights:" << std::endl;
	for each (auto weight_matrix in weights) {
		for each (auto weight_line in weight_matrix) {
			for each (auto weight in weight_line) {
				file << weight << " ";
			}
			file << std::endl;
		}
	}
	file.close();
	return true;
}

bool ANN::LoadData(
	std::string filepath,
	std::vector<std::vector<float>> & inputs,
	std::vector<std::vector<float>> & outputs)
{
	std::ifstream file(filepath);
	if (!file.is_open()) return false;
	const int CHAR_BUF_LEN = 100;
	char char_buffer[CHAR_BUF_LEN];
	file.getline(char_buffer, CHAR_BUF_LEN);
	std::string string_buffer = std::string(char_buffer);
	memset(char_buffer, 0, CHAR_BUF_LEN);
	if (string_buffer != std::string("input_count:"))
		throw "incorrect file format";
	int input_count;
	file >> input_count;
	file.getline(char_buffer, CHAR_BUF_LEN);
	file.getline(char_buffer, CHAR_BUF_LEN);
	string_buffer = std::string(char_buffer);
	memset(char_buffer, 0, CHAR_BUF_LEN);
	if (string_buffer != std::string("output_count:"))
		throw "incorrect file format";
	int output_count;
	file >> output_count;
	file.getline(char_buffer, CHAR_BUF_LEN);
	file.getline(char_buffer, CHAR_BUF_LEN);
	string_buffer = std::string(char_buffer);
	memset(char_buffer, 0, CHAR_BUF_LEN);
	if (string_buffer != std::string("primer_count:"))
		throw "incorrect file format";
	int primer_count;
	file >> primer_count;
	file.getline(char_buffer, CHAR_BUF_LEN);
	file.getline(char_buffer, CHAR_BUF_LEN);
	string_buffer = std::string(char_buffer);
	memset(char_buffer, 0, CHAR_BUF_LEN);
	if (string_buffer != std::string("data:"))
		throw "incorrect file format";
	inputs.resize(primer_count);
	outputs.resize(primer_count);
	//цикл по примерам
	for (int i = 0; i < primer_count; i++) {
		inputs[i].resize(input_count);
		//считываем входы
		for (int j = 0; j < input_count; j++) {
			file >> inputs[i][j];
		}
		file.getline(char_buffer, CHAR_BUF_LEN);
		//считываем выходы
		outputs[i].resize(output_count);
		for (int j = 0; j < output_count; j++) {
			file >> outputs[i][j];
		}
		file.getline(char_buffer, CHAR_BUF_LEN);
		file.getline(char_buffer, CHAR_BUF_LEN);
	}
	file.close();
	return true;
}

bool ANN::SaveData(
	std::string filepath,
	std::vector<std::vector<float>> & inputs,
	std::vector<std::vector<float>> & outputs)
{
	if (inputs.size() != outputs.size())
		throw "input size and output size must be the same";
	if (inputs.size() * outputs.size() == 0)
		throw "empty data";
	size_t input_count = inputs[0].size();
	size_t output_count = outputs[0].size();
	for (size_t i = 0; i < inputs.size(); i++) {
		if (inputs[i].size() != input_count)
			throw "incorrect input size";
		if (outputs[i].size() != output_count)
			throw "incorrect output size";
	}
	std::ofstream file(filepath);
	if (!file.is_open()) return false;
	file << std::setprecision(9);
	file << "input_count:" << std::endl;
	file << inputs[0].size() << std::endl;
	file << "output_count:" << std::endl;
	file << outputs[0].size() << std::endl;
	file << "primer_count:" << std::endl;
	file << inputs.size() << std::endl;
	file << "data:" << std::endl;
	for (size_t i = 0; i < inputs.size(); i++) {
		for (size_t j = 0; j < input_count; j++) {
			file << inputs[i][j] << "\t";
		}
		file << std::endl;
		for (size_t j = 0; j < output_count; j++) {
			file << outputs[i][j] << "\t";
		}
		file << std::endl;
		file << std::endl;
	}
	file.close();
	return true;
}

std::vector<size_t> ANN::ANeuralNetwork::GetConfiguration()
{
	return configuration;
}

float ANN::ANeuralNetwork::Activation(float neuronInput)
{
	if (activation_type == POSITIVE_SYGMOID) {
		return (1.f / (1.f + expf(-scale * neuronInput)));
	}
	else if (activation_type == BIPOLAR_SYGMOID) {
		return (2.f / (1.f + expf(-scale * neuronInput)) - 1.f);
	}
	return -1.f;
}

float ANN::ANeuralNetwork::ActivationDerivative(float activation)
{
	if (activation_type == POSITIVE_SYGMOID) {
		return scale * activation * (1.f - activation);
	}
	else if (activation_type == BIPOLAR_SYGMOID) {
		return scale * 0.5f * (1.f + activation) * (1.f - activation);
	}
	return -1;
}

ANN::ANeuralNetwork::~ANeuralNetwork()
{}

std::string ANN::GetTestString()
{
	return "You succesfully plug ANN library!";
}

void ANN::ANeuralNetwork::RandomInit()
{
	weights.resize(configuration.size() - 1);
	for (unsigned int layer_index = 0; layer_index < configuration.size() - 1; layer_index++) {
		weights[layer_index].resize(configuration[layer_index]);
		for (unsigned int from_index = 0; from_index < weights[layer_index].size(); from_index++) {
			weights[layer_index][from_index].resize(configuration[layer_index + 1]);
			for (unsigned int to_index = 0; to_index < weights[layer_index][from_index].size(); to_index++) {
				weights[layer_index][from_index][to_index] = 2.f * (rand() / float(RAND_MAX) - 0.5f);
			}
		}
	}
}

using namespace std;

ANNDLL_API float ANN::BackPropTraining(
	std::shared_ptr<ANN::ANeuralNetwork> ann,
	std::vector<std::vector<float>> & inputs,
	std::vector<std::vector<float>> & outputs,
	int maxIters,
	float eps,
	float speed,
	bool std_dump
)
{
	ann->RandomInit();
	if (inputs.size() != outputs.size())
		throw std::invalid_argument("invalid_argument");

	float currentError(0);
	int currentIter(0);

	do
	{
		currentError = 0;
		for (size_t countIdx = 0; countIdx < inputs.size(); countIdx++)
			currentError += BackPropTrainingIteration(ann, inputs[countIdx], outputs[countIdx], speed);

		currentIter++;
		currentError = sqrt(currentError);


		if (std_dump && currentIter % 1000 == 0)
		{
			cout << "Iteration: " << currentIter << "\tError: " << currentError << endl;
		}

		if (currentError < eps)
			ann->is_trained = true;

	} while (currentError > eps && currentIter <= maxIters);
	return currentError;
}

ANNDLL_API float ANN::BackPropTrainingIteration(
	std::shared_ptr<ANN::ANeuralNetwork> ann,
	const std::vector<float>& input,
	const std::vector<float>& output,
	float speed
)
{
	float currentError(0);

	vector<vector<float>> out(ann->configuration.size());
	//первый выход равен входу
	out[0] = input;

	//прямой ход
	for (size_t layerIdx = 0; layerIdx < ann->configuration.size() - 1; layerIdx++)
	{
		out[layerIdx + 1].resize(ann->configuration[layerIdx + 1]);
		for (size_t toIdx = 0; toIdx < ann->configuration[layerIdx + 1]; toIdx++)
		{
			out[layerIdx + 1][toIdx] = 0;
			for (size_t fromIdx = 0; fromIdx < ann->configuration[layerIdx]; fromIdx++)
			{
				out[layerIdx + 1][toIdx] += out[layerIdx][fromIdx] * ann->weights[layerIdx][fromIdx][toIdx];
			}
			out[layerIdx + 1][toIdx] = ann->Activation(out[layerIdx + 1][toIdx]);
		}
	}

	vector<vector<float>> sigma(ann->configuration.size());
	vector<vector<vector<float>>> dw(ann->configuration.size() - 1);
	sigma.back().resize(out.back().size());

	for (size_t layerIdx = 0; layerIdx < out.back().size(); layerIdx++)
	{
		sigma.back()[layerIdx] = (output[layerIdx] - out.back()[layerIdx]) *
			ann->ActivationDerivative(out.back()[layerIdx]);
		currentError += (output[layerIdx] - out.back()[layerIdx]) * (output[layerIdx] - out.back()[layerIdx]);
	}

	//обратный ход
	size_t layerIdx = ann->configuration.size() - 2;
	for (int i = ann->configuration.size() - 2; i >= 0; i--)
	{
		dw[layerIdx].resize(ann->weights[layerIdx].size());
		sigma[layerIdx].resize(ann->configuration[layerIdx], 0);

		for (size_t fromIdx = 0; fromIdx < ann->configuration[layerIdx]; fromIdx++)
		{
			for (size_t toIdx = 0; toIdx < ann->configuration[layerIdx + 1]; toIdx++)
			{
				sigma[layerIdx][fromIdx] += sigma[layerIdx + 1][toIdx] * ann->weights[layerIdx][fromIdx][toIdx];
			}

			sigma[layerIdx][fromIdx] *= ann->ActivationDerivative(out[layerIdx][fromIdx]);
			dw[layerIdx][fromIdx].resize(ann->weights[layerIdx][fromIdx].size());

			for (size_t toIdx = 0; toIdx < ann->configuration[layerIdx + 1]; toIdx++)
			{
				dw[layerIdx][fromIdx][toIdx] = speed * sigma[layerIdx + 1][toIdx] * out[layerIdx][fromIdx];
			}
		}
		layerIdx--;
	}

	//модификация весов
	for (size_t layerIdx = 0; layerIdx < ann->weights.size(); layerIdx++)
	{
		for (size_t fromIdx = 0; fromIdx < ann->weights[layerIdx].size(); fromIdx++)
		{
			for (size_t toIdx = 0; toIdx < ann->weights[layerIdx][fromIdx].size(); toIdx++)
			{
				ann->weights[layerIdx][fromIdx][toIdx] += dw[layerIdx][fromIdx][toIdx];
			}
		}
	}

	return currentError;
}