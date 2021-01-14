#pragma once
#include "Epoch.h"

namespace ga
{
	class GeneticAlgorithm
	{
	public:
		GeneticAlgorithm(std::vector<size_t> config, int population_size);

		virtual ~GeneticAlgorithm();

		pEpoch epoch;
	
		pEpoch Selection(double unchange_perc, double mutation_perc, double crossover_perc);
	private:
		int GetIndex(float val);
	};
}


