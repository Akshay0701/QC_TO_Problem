import numpy as np
import matplotlib.pyplot as plt
from dimod import BinaryQuadraticModel
from dwave.system import DWaveSampler, EmbeddingComposite

def generate_qubo_problem(size):
    # Define QUBO matrix (for simplicity, this is a random example)
    Q = np.random.rand(size, size) - 0.5
    return Q

def convert_to_qubo(Q):
    bqm = BinaryQuadraticModel('BINARY')
    
    for i in range(Q.shape[0]):
        for j in range(i + 1, Q.shape[1]):
            if Q[i, j] != 0:  # Only add non-zero interactions
                bqm.add_quadratic(i, j, Q[i, j])
                
    # Handle diagonal terms separately (as linear biases)
    for i in range(Q.shape[0]):
        if Q[i, i] != 0:  # Only add non-zero diagonal terms
            bqm.add_linear(i, Q[i, i])
            
    return bqm

def solve_qubo(bqm):
    sampler = EmbeddingComposite(DWaveSampler())
    response = sampler.sample(bqm, num_reads=1000)
    return response

def visualize_results(results):
    # Extract the best solution from results
    best_sample = results.first.sample
    result_values = [best_sample[i] for i in range(len(best_sample))]
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(result_values)), result_values)
    plt.xlabel('Variable Index')
    plt.ylabel('Value')
    plt.title('Quantum Annealing Results')
    plt.show()

# Example usage
size = 10  # Adjust based on your problem
Q = generate_qubo_problem(size)
bqm = convert_to_qubo(Q)
result = solve_qubo(bqm)
visualize_results(result)
