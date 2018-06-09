#include "Perceptron.hpp"
#include <iostream>





int main()
{
	Perceptron p;
	Perceptron* p1 = new Perceptron(2);
	Perceptron p2(2, 0.1, -0.9);

	assert(0.1 == p2.learnRate());

	std::vector<std::vector<int>> trainData;
	std::vector<int> expectedOutput;


	trainData.push_back(std::vector<int>{0, 0});
	trainData.push_back(std::vector<int>{0, 1});
	trainData.push_back(std::vector<int>{1, 0});
	trainData.push_back(std::vector<int>{1, 1});

	expectedOutput.push_back(0);
	expectedOutput.push_back(0);
	expectedOutput.push_back(0);
	expectedOutput.push_back(1);

	std::vector<double> test = { 0.0,0.5 };

	p1->train<int>(trainData.begin(), trainData.end(), expectedOutput.begin(), expectedOutput.end());
	assert(0 == p1->evaluate(test.begin(), test.end()));

	delete p1;

	return 0;
}