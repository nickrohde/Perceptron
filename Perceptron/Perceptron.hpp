#include "../../Utility/MasterInclude.hpp"
#include "../../Utility/utility.hpp"

#ifndef __PERCEPTRON_HPP
#define __PERCEPTRON_HPP


class Perceptron
{
public:
	Perceptron(void);
	Perceptron(const size_t ui_SIZE);
	Perceptron(const size_t ui_SIZE, const double d_LEARN_RATE, const double d_BIAS);
	

	template <typename T, typename Iter>
	void train(Iter start, Iter end, const int i_EXPECTED_OUTPUT)
	{
		//if (!b_initialized)
		//{
		//	initialize(trainData.size());
		//} // end if

		//if (trainData.size() != weights.size())
		//{
		//	throw std::invalid_argument("Instance must be same size as trained weights.");
		//} // end if

		bool b_changed = false;

		do
		{
			b_changed = false;

			for (auto i = 0; i < weights.size() && start != end; i++)
			{
				int temp = evaluate(start, end);

				if (temp != i_EXPECTED_OUTPUT)
				{
					weights[i] = weights[i] - learnRate() * (temp - i_EXPECTED_OUTPUT) * (*start);
					b_changed = true;
				} // end if
			} // end for
		} while (b_changed);
	} // end template train


	template <typename T>
	void train(const std::vector<std::vector<T>>& trainData, const std::vector<int> EXPECTED_OUTPUT, bool b_shuffleData = true)
	{
		indices.reserve(trainData.size());

		for (auto i = indices.size(); i < trainData.size(); i++)
		{
			indices.push_back(i);
		} // end for
		while (indices.size() > trainData.size())
		{
			indices.pop_back();
		} // end while

		if (b_shuffleData)
		{
			std::shuffle(indices.begin(), indices.end(), std::mt19937());
		} // end if

		for (auto i = 0; i < trainData.size(); i++)
		{
			train<T>(trainData[indices[i]].begin(), trainData[indices[i]].end(), EXPECTED_OUTPUT[indices[i]]);
		} // end for
	} // end template train


	template <typename Iter>
	int evaluate(Iter start, Iter end) 
	{
		//if (instance.size() != weights.size())
		//{
		//	throw std::invalid_argument("Instance must be same size as trained weights.");
		//} // end if
		
		double d_sum = 0.0;

		for (size_t i = 0; i < weights.size() && start != end; i++, start++)
		{
			d_sum += (*start * weights[i]);
		} // end for

		d_sum += d_bias;

		return (d_sum > 0) ? (1) : (0);
	} // end template evaluate


	inline void clearWeights(void) noexcept { b_initialized = false; }

	inline double learnRate(void) noexcept { return d_learningRate; }

	const double DEFAULT_LEARN_RATE = 0.4;
	const double DEFAULT_BIAS = -1.0;

protected:
	std::vector<double> weights;
	bool b_initialized;
	double d_learningRate;
	double d_bias;

private:
	std::vector<size_t> indices;

	void initialize(const size_t ui_SIZE);
};


#endif
