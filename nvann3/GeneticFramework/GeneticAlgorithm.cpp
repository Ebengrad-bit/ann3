#include <algorithm>
#include "GeneticAlgorithm.h"

using namespace ga;
using namespace std;

GeneticAlgorithm::GeneticAlgorithm(vector<size_t> config, int population_size)
{
	vector<pair<int, pIIndividual>> new_population;
	for (int i = 0; i < population_size; i++)
	{
		Individual newIn(config, ANN::ANeuralNetwork::ActivationType::POSITIVE_SYGMOID, 1.0f, "xor.data");
		pIIndividual new_individual = std::make_shared<Individual>(newIn);
		new_population.push_back(pair<int, pIIndividual>(0, new_individual));
	}
	Epoch newEpoch;
	this->epoch = make_shared<Epoch>(newEpoch);
	this->epoch->population = new_population;
}


GeneticAlgorithm::~GeneticAlgorithm()
{
	this->epoch->population.clear();
}

int GeneticAlgorithm::GetIndex(float val)
{
	float last = 0;
	for (int i = this->epoch->population.size()-1; i>=0; i--) {
		if (last <= val && val >= this->epoch->population[i].first) return i;
		else last = this->epoch->population[i].first;
	}
	return 0;
}

pEpoch GeneticAlgorithm::Selection(double unchange_perc, double mutation_perc, double crossover_perc)
{
	sort(epoch->population.begin(), epoch->population.end(),
		[](pair<int, pIIndividual> a, pair<int, pIIndividual> b)
	{ return a.first > b.first; });

	vector<pair<int, pIIndividual>> new_population;
	int survivors_count = (int)(epoch->population.size() * unchange_perc);
	for (int i = 0; i < survivors_count; i++)
	{
		new_population.push_back(epoch->population[i]);
		new_population[i].first = 0;
	}
	std::random_device rd; std::mt19937 randm(rd());

	int val = epoch->population[0].first - epoch->population[epoch->population.size() - 1].first;
	int min = epoch->population[epoch->population.size() - 1].first;
	for (int j = 0; j < (int)(epoch->population.size()* crossover_perc); j++)
	{
		int p1 = 0;	int p2 = 0;
		while (p1 == p2)
		{
			p1 = GetIndex(min + rand() % val);
			p2 = GetIndex(min + rand() % val);
		}
		pIIndividual childe = epoch->population[p1].second->Crossover(epoch->population[p2].second);
		new_population.push_back(pair<int, pIIndividual>(0, childe));
	}
	for (int k = 0; k < (int)(epoch->population.size() * mutation_perc); k++)
	{
		int r = randm() % survivors_count;
		pIIndividual mutant = new_population[r].second->Mutation();
		new_population.push_back(pair<int, pIIndividual>(0, mutant));
	}
	if (new_population.size() <= 0)
		new_population.push_back(this->epoch->population[0]);
	this->epoch->population = new_population;
	return this->epoch;
}
