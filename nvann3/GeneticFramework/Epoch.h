#pragma once
#include "Individual.h"
namespace ga
{

	class Epoch
	{
	public:

		Epoch();

		virtual ~Epoch();


		std::vector<std::pair<int, pIIndividual>> population;


		void EpochBattle();
	};

	typedef std::shared_ptr<Epoch> pEpoch;
}