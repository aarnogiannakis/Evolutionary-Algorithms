# Evolutionary-Algorithms

This script implements an Evolutionary Algorithm (EA) designed to address a complex multi-processor job scheduling problem. 
The primary objective is to minimize the makespan by efficiently allocating jobs across multiple processors, 
while considering the duration and processor assignment constraints for each job operation.

The EA framework incorporates the following key strategies and components:

  - **Initialization**: Generates an initial population of feasible solutions using a permutation-based heuristic 
                  that respects processor constraints and provides a diverse starting point for the evolutionary process.

  - **Crossover Operators**: Implements the Partially Mapped Crossover (PMX) method to combine pairs of parent solutions, 
                       preserving relative job orderings while introducing new offspring into the population.

  **Mutation Operators**: A random swap mutation operator is employed to introduce diversity by exchanging the positions of two jobs within a solution, 
                      enabling the exploration of new areas in the solution space.

  **Selection Mechanism: **A tournament selection process is used to choose parent solutions based on their makespan, 
                       ensuring that better-performing solutions have a higher probability of producing offspring.

  Population Management: The algorithm dynamically maintains a diverse population by replacing weaker solutions with better-performing offspring, 
                          thus balancing exploration and exploitation.

  Adaptive Mechanism: An adaptive adjustment mechanism is employed to fine-tune the probabilities of applying crossover and mutation operators 
                      based on their success in reducing the makespan, fostering a balance between solution diversification and intensification.

  Acceptance Criterion: Utilizes an elitist strategy where the best solutions are always retained in the population, 
                        ensuring that the global best solution found throughout the process is preserved.

  Termination Criteria: The algorithm concludes after a predefined number of iterations or when a specified computational time limit is reached, 
                        ensuring computational efficiency while striving for optimal solutions.

  Objective Calculation: The primary objective is to minimize the makespan, which is calculated as the completion time of the last job across all processors. 
                         This ensures that the schedule is optimized for overall efficiency.
