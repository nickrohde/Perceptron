#include "Perceptron.hpp"

Perceptron::Perceptron(void)
{
	b_initialized = false;
	d_learningRate = DEFAULT_LEARN_RATE;
	d_bias = DEFAULT_BIAS;
} // end Default Constructor

Perceptron::Perceptron(const size_t ui_SIZE) : Perceptron()
{
	initialize(ui_SIZE);
} // end Constructor 1

Perceptron::Perceptron(const size_t ui_SIZE, const double d_LEARN_RATE, const double d_BIAS) : Perceptron(ui_SIZE)
{
	d_learningRate = d_LEARN_RATE;
	d_bias = d_BIAS;
} // end Constructor 2

void Perceptron::initialize(const size_t ui_SIZE)
{
	b_initialized = true;
	weights.clear();
	weights.reserve(ui_SIZE);

	for (auto i = 0; i < ui_SIZE; i++)
	{
		weights.push_back(getRandomRealInRange<double>(0.0, 1.0));
	} // end for
} // end method initialize
