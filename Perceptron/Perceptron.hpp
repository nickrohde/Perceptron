#ifndef __PERCEPTRON_HPP
#define __PERCEPTRON_HPP

#include "../../Utility/MasterInclude.hpp"
#include "../../Utility/utility.hpp"
#include <cerrno>


///<summary>Errors that could occur during Perceptron execution; ERRNO will be set to the appropriate values.</summary>
enum PerceptronErrors
{
	///<summary>Errno will be set to this value should memory allocation fail.</summary>
	OUT_OF_MEMORY = 1,
};


///<summary>
///			Perceptron learning algorithm. 
///			Evaluates inputs to either 0 or 1 based on learned weights. The value returned by Perceptron::evaluate is 
///			random until the perceptron has been trained.
///
///			Inputs may be in any collection that supports iterators for both training and evaluation.
///</summary>
class Perceptron
{

	#pragma region Public:
	public:

		#pragma region Constructor:
			///<summary>Default Constructor for Perceptron. Constructs uninitialized Perceptron with default values.</summary>
			///<remarks>Complexity: O(1)</remarks>
			Perceptron(void);


			///<summary>Constructor that sets the number of inputs for the Perceptron.</summary>
			///<param name="ui_SIZE">The number of inputs to set for this Perceptron.</param>
			///<remarks>Complexity: O(n)</remarks>
			inline Perceptron(const size_t ui_SIZE) : Perceptron() { initialize(ui_SIZE); }


			///<summary>Constructor that sets the number of inputs, learn rate, and bias for the Perceptron.</summary>
			///<param name="ui_SIZE">The number of inputs to set for this Perceptron.</param>
			///<param name="d_LEARN_RATE">The learn rate to use.</param>
			///<param name="d_BIAS">The bias value to use.</param>
			///<remarks>Complexity: O(n)</remarks>
			Perceptron(const size_t ui_SIZE, const double d_LEARN_RATE, const double d_BIAS);


			///<summary>Copy constructor operator.</summary>
			///<param name="OTHER">Perceptron to copy.</param>
			///<remarks>Complexity: O(n)</remarks>
			inline Perceptron(const Perceptron& OTHER) { *this = OTHER; }


			///<summary>Move constructor operator.</summary>
			///<param name="other">Perceptron to move.</param>
			///<remarks>Complexity: O(1)</remarks>
			inline Perceptron(Perceptron&& other) noexcept { *this = other; }

		#pragma endregion

		#pragma region Operations:
			///<summary>Trains this neuron with the given input set.</summary>
			///<typeparam name="T">Some type that allows multiplication with doubles.</typeparam>
			///<typeparam name="Iter">Non-const Iterator type.</typeparam>
			///<param name="start">Iterator to the start of the input.</param>
			///<param name="end">Iterator to the end of the input.</param>
			///<param name="i_EXPECTED_OUTPUT">The expected output for this input.</param>
			///<exception name="std::invalid_argument">
			///	Thrown if the number of inputs differs from the number of weights. Only thrown if the perceptron was previously initialized.
			///	May also be thrown if memory allocation failed, check ERRNO.
			///	</exception>
			///<remarks>Train may be called on uninitialized objects and it will initialize the weights appropriately.
			/// Complexity: O(n)
			///</remarks>
			template <typename T, typename Iter>
			void train(Iter start, Iter end, const int i_EXPECTED_OUTPUT)
			{
				auto size = std::distance(start, end);
				bool b_changed = false;

				if (!b_initialized)
				{
					initialize(size);
				} // end if

				if (size != weights.size())
				{
					throw std::invalid_argument("Instance must be same size as trained weights.");
				} // end if

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


			///<summary>Bulk trains this neuron with the given input sets.</summary>
			///<typeparam name="T">Some type that allows multiplication with doubles.</typeparam>
			///<typeparam name="Iter1">Non-const Iterator type for container of containers.</typeparam>
			///<typeparam name="Iter2">Non-const Iterator type for container of ints.</typeparam>
			///<param name="train_data_start">Iterator to the start of the input.</param>
			///<param name="train_data_end">Iterator to the end of the input.</param>
			///<param name="output_start">Iterator to the start of the output.</param>
			///<param name="output_end">Iterator to the end of the output.</param>
			///<param name="b_shuffleData">Whether or not to shuffle the data for training.</param>
			///<exception name="std::invalid_argument">Thrown if the number of iputs differs from the number of outputs.</exception>
			///<remarks>Train may be called on uninitialized objects and it will initialize the weights appropriately.
			/// Complexity: O(n*m) for n inputs and m training instances.
			///</remarks>
			template <typename T, typename Iter1, typename Iter2>
			void train(Iter1 train_data_start, Iter1 train_data_end, Iter2 output_start, Iter2 output_end, bool b_shuffleData = true)
			{
				auto output_size = std::distance(output_start, output_end);
				auto input_size = std::distance(train_data_start, train_data_end);

				try
				{
					indices.reserve(input_size);
				} // end try
				catch (std::bad_alloc)
				{
					errno = PerceptronErrors::OUT_OF_MEMORY;
					return;
				} // end catch

				if (input_size != output_size)
				{
					throw std::invalid_argument("Input size and output size must match!");
				} // end if

				for (auto i = indices.size(); i < input_size; i++)
				{
					indices.push_back(i);
				} // end for
				while (indices.size() > input_size)
				{
					indices.pop_back();
				} // end while

				if (b_shuffleData)
				{
					std::shuffle(indices.begin(), indices.end(), std::mt19937());
				} // end if

				for (auto i = 0; train_data_start != train_data_end; i++)
				{
					train<T>((train_data_start + indices[i]).begin(), (train_data_start + indices[i]).end(), *(output_start + indices[i]));
				} // end for
			} // end template train


			///<summary>Evaluates the Perceptron with the given input.</summary>
			///<typeparam name="Iter">Non-const Iterator type.</typeparam>
			///<param name="start">Iterator to the start of the input.</param>
			///<param name="end">Iterator to the end of the input.</param>
			///<returns>The output for this input (0 or 1).</returns>
			///<exception name="std::invalid_argument">Thrown if the number of iputs differ from the number of weights.</exception>
			///<remarks>Complexity: O(n)</remarks>
			template <typename Iter>
			int evaluate(Iter start, Iter end) 
			{
				auto size = std::distance(start, end);
				double d_sum = 0.0;

				if (size != weights.size())
				{
					throw std::invalid_argument("Instance must be same size as trained weights.");
				} // end if

				for (size_t i = 0; i < weights.size() && start != end; i++, start++)
				{
					d_sum += (*start * weights[i]);
				} // end for

				d_sum += d_bias;

				return (d_sum > 0) ? (1) : (0);
			} // end template evaluate


			///<summary>Clears the weights of this Perceptron and returns it to uninitialized state.</summary>
			///<remarks>Complexity: O(1)</remarks>
			inline void clearWeights(void) noexcept { b_initialized = false; }

		#pragma endregion

		#pragma region Accessors:
			///<summary>Getter for the learn rate.</summary>
			///<returns>The current learn rate of this Perceptron.</returns>
			///<remarks>Complexity: O(1)</remarks>
			inline double learnRate(void) const noexcept { return d_learningRate; }


			///<summary>Getter for the bias value.</summary>
			///<returns>The current bias value of this Perceptron.</returns>
			///<remarks>Complexity: O(1)</remarks>
			inline double bias(void) const noexcept { return d_bias; }


			///<summary>Getter for the number of inputs.</summary>
			///<returns>The current number of inputs accepted by this Perceptron.</returns>
			///<remarks>Complexity: O(1)</remarks>
			inline size_t size(void) const noexcept { return (b_initialized) ? (weights.size()) : (0); }


			///<summary>Getter for the weight of a specific input.</summary>
			///<param name="ui_INDEX">Index of the weight to retrieve.</param>
			///<returns>The weight associated with the specified input.</returns>
			///<exception name="std::out_of_range">Thrown if <paramref name="ui_INDEX"/> is not valid or the object is uninitialized at point of call.</exception>
			///<remarks>Complexity: O(1)</remarks>
			inline double weight(const size_t ui_INDEX) const { if (ui_INDEX >= size() || !b_initialized) throw std::out_of_range("Index out of range!"); else return weights[ui_INDEX]; }


			///<summary>Getter for all weights.</summary>
			///<returns>A copy of the weights of this Perceptron.</returns>
			///<exception name="std::logic_error">Thrown if the object is uninitialized at point of call.</exception>
			///<remarks>Complexity: O(n)</remarks>
			std::vector<double>& weight(void) const;


			///<summary>Whether or not this perceptron is initialized.</summary>
			///<returns>True iff the perceptron has been initialized, otherwise false.</returns>
			///<remarks>Complexity: O(1)</remarks>
			inline bool isInitialized(void) const noexcept { return b_initialized; }

		#pragma endregion

		#pragma region Mutators:
			///<summary>Setter for the bias value.</summary>
			///<param name="d_NEW_BIAS">New bias value to use.</param>
			///<remarks>Complexity: O(1)</remarks>
			inline void setBias(const double d_NEW_BIAS) noexcept { d_bias = d_NEW_BIAS; }


			///<summary>Setter for the learn rate.</summary>
			///<param name="d_NEW_LR">New learn rate to use.</param>
			///<remarks>Complexity: O(1)</remarks>
			inline void setLearnRate(const double d_NEW_LR) noexcept { d_learningRate = d_NEW_LR; }


			///<summary>Setter for the weight of a specific input.</summary>
			///<param name="ui_INDEX">Index of weight to change.</param>
			///<exception name="std::out_of_range">Thrown if <paramref name="ui_INDEX"/> is not valid.</exception>
			///<remarks>Complexity: O(1)</remarks>
			inline void setWeight(const size_t ui_INDEX, const double d_VALUE) { if (ui_INDEX >= size()) throw std::out_of_range("Index out of range!"); else weights[ui_INDEX] = d_VALUE; }


			///<summary>Setter for all weights.</summary>
			///<param name="values">New weights to use.</param>
			///<remarks>Does nothing if number of new weights does not match number of old weights. Complexity: O(n)</remarks>
			inline void setWeight(std::vector<double>& values) { if(values.size() == weights.size() || !b_initialized) std::copy(values.begin(), values.end(), weights.begin()); }


			///<summary>Setter for all weights.</summary>
			///<param name="values">New weights to use.</param>
			///<remarks>Does nothing if number of new weights does not match number of old weights. Complexity: O(1)</remarks>
			inline void setWeight(std::vector<double>&& values) { if (values.size() == weights.size() || !b_initialized) weights = std::move(values); }

		#pragma endregion

		#pragma region Operators:
			///<summary>Copy assignment operator.</summary>
			///<param name="OTHER">Perceptron to copy.</param>
			///<returns>Reference to this perceptron.</returns>
			///<remarks>Complexity: O(n)</remarks>
			Perceptron & operator=(const Perceptron& OTHER);
		

			///<summary>Move assignment operator.</summary>
			///<param name="other">Perceptron to move.</param>
			///<returns>Reference to this perceptron.</returns>
			///<remarks>Complexity: O(1)</remarks>
			Perceptron & operator=(Perceptron&& other) noexcept;

		#pragma endregion

		#pragma region Constants:
			///<summary>The default learning rate.</summary>
			const double DEFAULT_LEARN_RATE = 0.4;


			///<summary>The default bias value.</summary>
			const double DEFAULT_BIAS = -1.0;

		#pragma endregion

	#pragma endregion

	#pragma region Protected:
	protected:

		#pragma region Data Members:
			///<summary>Weights of the Perceptron.</summary>
			std::vector<double> weights;

			///<summary>The learning rate of the Perceptron.</summary>
			double d_learningRate;

			///<summary>The bias of the Perceptron.</summary>
			double d_bias;
		#pragma endregion

	#pragma endregion

	#pragma region Private:
	private:
	
		#pragma region Data Members:
			///<summary>Indices for shuffling the input.</summary>
			std::vector<size_t> indices;

			///<summary>Whether or not the Perceptron is initialized.</summary>
			bool b_initialized;

		#pragma endregion

		#pragma region Operations:
			///<summary>Initializes the weights.</summary>
			///<param name="ui_SIZE">Number of weights to create.</param>
			///<remarks>Complexity: O(n)</remarks>
			void initialize(const size_t ui_SIZE);

		#pragma endregion

	#pragma endregion

}; // end Class Perceptron


#endif
