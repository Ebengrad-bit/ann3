#include <algorithm>
#include <iostream>
#include "GeneticAlgorithm.h"
using  namespace std;
using namespace ga;

int main()
{
	shortcut:
	cout << "size: "; int population_size = 20; cin >> population_size;
	if (population_size > 0)
	{
		vector<size_t> config;
		// Neural network configuration:
		config.push_back(2); config.push_back(4); config.push_back(1);
		GeneticAlgorithm GAl = GeneticAlgorithm(config, population_size);

		cout << "Survivor%: "; float unchange_perc = 50;  cin >> unchange_perc;
		cout << "Crossover%: "; float crossover_perc = 30; cin >> crossover_perc;
		cout << "Mutation%: "; float mutation_perc = 20; cin >> mutation_perc;
		unchange_perc *= 0.01; crossover_perc *= 0.01; mutation_perc *= 0.01;
		cout << "Epoh limit: "; int lineage = 1000;  cin >> lineage;
		cout << endl << "Evolve or die!" << endl;
		int iter = 0;
		int nom = 0;
		while ((lineage > 0) && (population_size > 1))
		{
			GAl.epoch->EpochBattle();
			nom++;
			if ((iter >= 20) || (lineage <= 1))
			{
				cout << endl << "War #" << nom << " points:" << endl;
				for each (auto unit in GAl.epoch->population) { cout << unit.first << '\t'; }
			}
			GAl.Selection(unchange_perc, mutation_perc, crossover_perc);
			if (population_size - GAl.epoch->population.size() < 0)
			{
				cout << "Parameters lead to overflow! Try again" << endl;
				goto shortcut;
			}
			if ((iter >= 20) || (lineage <= 1))
			{
				cout << endl << "Selection results: ";
				cout << (int)(population_size * unchange_perc) << " Evolved, ";
				cout << (int)(population_size * mutation_perc) << " Mutated, ";
				cout << (int)(population_size * crossover_perc) << " Sons of their parents, ";
				cout << (population_size - GAl.epoch->population.size()) << " Dead" << endl;
				iter = 0;
			}
			population_size = GAl.epoch->population.size();
			lineage--;
			iter++;
		}

		if (lineage > 0) { cout << "Wars ended! " << lineage << " more wars never be made." << endl; }

		GAl.epoch->EpochBattle(); 
		sort(GAl.epoch->population.begin(), GAl.epoch->population.end(),
			[](pair<int, pIIndividual> a, pair<int, pIIndividual> b)
				{ return a.first > b.first; });

		vector<vector<float>> inputs, outputs;
		ANN::LoadData("xor.data", inputs, outputs);
		cout << "Need: ";
		for (int i = 0; i <= 1; i++)
			for (int j = 0; j <= 1; j++)
				cout << i << "^" << j << "= " << (i^j) << "  ";
		cout << endl << "Total war results:" << endl << "Score\tPredictions";
		for (int i = 0; i < population_size; i++)
		{
			cout << endl << GAl.epoch->population[i].first;
			float eps = 0.0f;
			for (int k = 0; k < inputs.size(); k++)
			{
				auto prediction = GAl.epoch->population[i].second->MakeDecision(inputs[k]);
				for (int l = 0; l < outputs[k].size(); l++) { cout << "  " << inputs[k][0] << "^" << inputs[k][1] << "= "<< prediction[l]; 
					eps += (outputs[k][l] - prediction[l]) * (outputs[k][l] - prediction[l]); 
				}
			}
			cout << "\te=" << eps << endl;
		}
	}
	cout << endl;
	return 0;
}