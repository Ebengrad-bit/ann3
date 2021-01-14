#pragma once
#include <random>
#include <ANN.h>
#include "IIndividual.h"

namespace ga
{
	class Individual : public IIndividual, public ANN::ANeuralNetwork
	{
	public:
		Individual(
			std::vector<size_t> & configuration,
			ANN::ANeuralNetwork::ActivationType activation_type,
			float scale,
			std::string data_source
		);
		~Individual();


		std::shared_ptr<IIndividual> Mutation();


		std::shared_ptr<IIndividual> Crossover(std::shared_ptr<IIndividual> individual);


		std::pair<int, int> Spare(std::shared_ptr<IIndividual> individual);


		std::vector<float> MakeDecision(std::vector<float> & input);


		std::shared_ptr<IIndividual> Clone();

		virtual std::string GetType() override;

		virtual std::vector<float> Predict(std::vector<float> & input) override;

	protected:

		std::vector<std::vector<float>> inputs, outputs;
	};
};