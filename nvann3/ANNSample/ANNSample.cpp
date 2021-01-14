#include <iostream>
#include <NeuralRealisation.h>
using namespace std;
using namespace ANN;

int main()
{
	cout << "hello ANN!" << endl;
	cout << GetTestString().c_str() << endl;
	std::vector<size_t> l;
	l.push_back(3);
	l.push_back(3);
	l.push_back(3);
	auto ann = CreateNeuralNetwork(l, ANN::ANeuralNetwork::ActivationType::POSITIVE_SYGMOID, 1);
	ann->Load("ann.txt");
	return 0;
}