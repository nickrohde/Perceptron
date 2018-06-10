#include "Perceptron.hpp"

Perceptron::Perceptron(void)
{
	b_initialized = false;
	d_learningRate = DEFAULT_LEARN_RATE;
	d_bias = DEFAULT_BIAS;
} // end Default Constructor


Perceptron::Perceptron(const size_t ui_SIZE, const double d_LEARN_RATE, const double d_BIAS) : Perceptron(ui_SIZE)
{
	// Perceptron::Perceptron(size_t) will initialize weights
	d_learningRate = d_LEARN_RATE;
	d_bias = d_BIAS;
} // end Constructor 3


std::vector<double>* Perceptron::weight(void) const
{
	if (!b_initialized) // ensure object is in valid state
	{
		throw std::logic_error("Object is not initialized.");
	} // end if

	auto other = new std::vector<double>();

	try
	{
		other->reserve(size());
	} // end try
	catch (std::bad_alloc)
	{
		errno = PerceptronErrors::OUT_OF_MEMORY;
		other->clear();
		return nullptr;
	} // end catch

	// copy weights into output vector
	std::copy(weights.begin(), weights.end(), other->begin());

	return other;
} // end method weight


Perceptron & Perceptron::operator=(const Perceptron & OTHER)
{
	b_initialized = OTHER.b_initialized;
	d_bias = OTHER.bias();
	d_learningRate = OTHER.learnRate();

	try
	{
		weights.reserve(OTHER.size());
	} // end try
	catch (std::bad_alloc)
	{
		errno = PerceptronErrors::OUT_OF_MEMORY;
		return *this;
	} // end catch

	std::copy(OTHER.weights.begin(), OTHER.weights.end(), weights.begin());

	return *this;
} // end Copy Assignment


Perceptron & Perceptron::operator=(Perceptron && other) noexcept
{
	b_initialized = other.b_initialized;
	d_bias = other.bias();
	d_learningRate = other.learnRate();

	weights = std::move(other.weights);

	return *this;
} // end Move Assignment


void Perceptron::initialize(const size_t ui_SIZE)
{
	weights.clear();

	try
	{
		weights.reserve(ui_SIZE);
	} // end try
	catch (std::bad_alloc) // ensure memory allocation succeeded
	{
		errno = PerceptronErrors::OUT_OF_MEMORY;
		return;
	} // end catch

	// initialize weights to random values between 0 and 1
	for (auto i = 0; i < ui_SIZE; i++)
	{
		weights.push_back(getRandomRealInRange<double>(0.0, 1.0));
	} // end for

	b_initialized = true;
} // end method initialize
