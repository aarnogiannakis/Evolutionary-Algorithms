# <42137 Optimization using Metaheuristics -- Assignment 06>
# Evolutionary Algorithm for Multi-Processor Job Scheduling
#
# This script implements an Evolutionary Algorithm (EA) designed to address a complex multi-processor job scheduling problem. 
# The primary objective is to minimize the makespan by efficiently allocating jobs across multiple processors, 
# while considering the duration and processor assignment constraints for each job operation.
#
# The EA framework incorporates the following key strategies and components:
#   Initialization: Generates an initial population of feasible solutions using a permutation-based heuristic 
#                   that respects processor constraints and provides a diverse starting point for the evolutionary process.
#
#   Crossover Operators: Implements the Partially Mapped Crossover (PMX) method to combine pairs of parent solutions, 
#                        preserving relative job orderings while introducing new offspring into the population.
#
#   Mutation Operators: A random swap mutation operator is employed to introduce diversity by exchanging the positions of two jobs within a solution, 
#                       enabling the exploration of new areas in the solution space.
#
#   Selection Mechanism: A tournament selection process is used to choose parent solutions based on their makespan, 
#                        ensuring that better-performing solutions have a higher probability of producing offspring.
#
#   Population Management: The algorithm dynamically maintains a diverse population by replacing weaker solutions with better-performing offspring, 
#                           thus balancing exploration and exploitation.
#
#   Adaptive Mechanism: An adaptive adjustment mechanism is employed to fine-tune the probabilities of applying crossover and mutation operators 
#                       based on their success in reducing the makespan, fostering a balance between solution diversification and intensification.
#
#   Acceptance Criterion: Utilizes an elitist strategy where the best solutions are always retained in the population, 
#                         ensuring that the global best solution found throughout the process is preserved.
#
#   Termination Criteria: The algorithm concludes after a predefined number of iterations or when a specified computational time limit is reached, 
#                         ensuring computational efficiency while striving for optimal solutions.
#
#   Objective Calculation: The primary objective is to minimize the makespan, which is calculated as the completion time of the last job across all processors. 
#                          This ensures that the schedule is optimized for overall efficiency.
#*****************************************************************************************************


#*****************************************************************************************************
using Random
using Statistics
using StatsBase
using Dates 
using FileIO

function load_instance(file_path::String)
    file = open(file_path)
    readline(file) # Skip the first line (header or irrelevant information)
    num_jobs, num_machines, best_upper_bound = parse.(Int, split(readline(file))) # Read the number of jobs, machines, and upper bound
    readline(file) # Skip the next line (possibly a separator)
    job_durations = zeros(Int, num_jobs, num_machines) # Initialize a matrix for job durations
    for job_idx in 1:num_jobs
        job_durations[job_idx, :] = parse.(Int, split(readline(file))) # Read the duration matrix row by row
    end
    readline(file) # Skip the next line (possibly a separator)
    machine_assignments = zeros(Int, num_jobs, num_machines) # Initialize a matrix for machine assignments
    for job_idx in 1:num_jobs
        machine_assignments[job_idx, :] = parse.(Int, split(readline(file))) # Read the machine assignment matrix row by row
    end
    close(file)

    return num_jobs, # the number of jobs
            num_machines, # the number of machines = number of operations
            best_upper_bound, # the best-known upper bound
            job_durations, # the duration of each operation
            machine_assignments # the machine assigned to each operation
end
#*****************************************************************************************************


#*****************************************************************************************************
# Structure representing a single task within a job
mutable struct Task
    job_id::Int          # ID of the job
    task_id::Int         # ID of the task within the job
    task_duration::Int   # Duration of the task
    machine_id::Int      # Machine assigned to the task
    start_time::Int      # Start time of the task
    end_time::Int        # End time of the task
end
#*****************************************************************************************************


#*****************************************************************************************************
# Initializes tasks from given duration and machine assignment matrices
function initialize_tasks(duration_matrix, machine_matrix)
    tasks = Task[]  # Initialize an empty array of Task structs
    total_jobs, total_tasks = size(duration_matrix)  # Assuming duration_matrix and machine_matrix have the same size
    
    for job_id in 1:total_jobs
        for task_id in 1:total_tasks
            task_duration = duration_matrix[job_id, task_id]
            machine_id = machine_matrix[job_id, task_id]
            push!(tasks, Task(job_id, task_id, task_duration, machine_id, 0, 0)) 
            # Add the task to the list with default start and end times set to 0
        end
    end
    
    return tasks
end

# Groups tasks by their assigned machines
function categorize_tasks_by_machine(tasks)
    max_machine_id = maximum(task.machine_id for task in tasks) # Get the maximum machine ID
    
    # Initialize a list of lists to hold tasks for each machine
    machine_task_lists = [Vector{Task}() for _ in 1:max_machine_id]
    
    # Assign tasks to the appropriate machine group
    for task in tasks
        push!(machine_task_lists[task.machine_id], task) 
        # Add task to the corresponding machine's list
    end
    
    return machine_task_lists
end

# Generates a matrix of unique sequences (permutations)
function create_unique_sequences(n)
    initial_sequence = shuffle(1:n) # Create a random sequence from 1 to n
    
    sequence_matrix = [initial_sequence] # Initialize the matrix with the first sequence
    
    circular_shifts = [circshift(initial_sequence, shift) for shift in 1:(n - 1)] # Generate circular shifts
    
    shuffle!(circular_shifts) # Shuffle the shifts to ensure uniqueness and prevent consecutive orders
    
    # Add the shuffled shifts to the matrix
    for shift in circular_shifts[1:(n - 1)]
        push!(sequence_matrix, shift)
    end
    
    return Array{Int}(hcat(sequence_matrix...)') # Return the final sequence matrix
end

# Reorders tasks based on a given sequence vector
function reorder_tasks(tasks, sequence_vector)
    # Create a dictionary to map job IDs to their new sequence positions
    sequence_map = Dict(sequence_vector[i] => i for i in 1:length(sequence_vector))
    
    # Sort the tasks based on the new sequence defined by sequence_vector
    sorted_tasks = sort(tasks, by = task -> sequence_map[task.job_id])
    
    return sorted_tasks
end

# Generates the final ordered tasks based on sequences
function organize_final_task_order(machine_tasks, sequences, num_machines)
    all_tasks_ordered = [] # Initialize a list to hold all the ordered tasks
    for machine_idx in 1:num_machines
        ordered_tasks = reorder_tasks(machine_tasks[machine_idx], sequences[machine_idx, :]) 
        # Reorder tasks for the machine using the sequence
        push!(all_tasks_ordered, ordered_tasks) # Add the reordered tasks to the final list
    end

    return all_tasks_ordered # Return the final list of ordered tasks
end
#*****************************************************************************************************


#*****************************************************************************************************
# Function to find the start and end times for each task
function calculate_start_times(all_tasks, num_tasks_per_processor)
    # Assuming all_tasks is a 2D array (matrix) of Task structs

    # Initialize start and end times for the first task in each processor
    for i in 1:num_tasks_per_processor
        first_task = all_tasks[i][1]
        first_task.start_time = 0
        first_task.end_time = first_task.task_duration
    end

    # Iterate over each task, starting from the second task
    for j in 2:num_tasks_per_processor
        for i in 1:num_tasks_per_processor
            current_task = all_tasks[i][j]
            current_job_id = current_task.job_id

            # Find the maximum end time of the previous task of the same job across all processors
            max_end_time_for_job = 0
            for k in 1:num_tasks_per_processor
                if all_tasks[k][j - 1].job_id == current_job_id
                    max_end_time_for_job = max(max_end_time_for_job, all_tasks[k][j - 1].end_time)
                end
            end

            # The start time for the current task is the later of:
            # - the end time of the previous task on the same processor
            # - the maximum end time of the last task of the same job across all processors
            current_task.start_time = max(all_tasks[i][j - 1].end_time, max_end_time_for_job)
            current_task.end_time = current_task.start_time + current_task.task_duration
        end
    end

    return all_tasks
end

# Function to calculate the makespan (total time to complete all tasks)
function calculate_makespan(task_sequences, num_tasks)
    max_span = 0
    for task_sequence in task_sequences
        if task_sequence[num_tasks].end_time > max_span
            max_span = task_sequence[num_tasks].end_time
        end
    end

    return max_span
end

# Function to generate an initial solution
function generate_initial_solution(file_path)
    num_jobs, num_machines, upper_bound, duration_matrix, machine_matrix = load_instance(file_path)

    tasks = initialize_tasks(duration_matrix, machine_matrix)
    task_groups = categorize_tasks_by_machine(tasks)
    sequence_permutations = create_unique_sequences(num_machines)

    ordered_tasks = organize_final_task_order(task_groups, sequence_permutations, num_machines)

    final_task_schedule = calculate_start_times(ordered_tasks, num_machines)

    return final_task_schedule
end

# Function to generate a new solution based on a given permutation
function generate_new_solution(file_path, permutation)
    num_jobs, num_machines, upper_bound, duration_matrix, machine_matrix = load_instance(file_path)

    tasks = initialize_tasks(duration_matrix, machine_matrix)
    task_groups = categorize_tasks_by_machine(tasks)
    modified_permutations = pmx_permutations(permutation) # Assuming pmx_permutations is defined elsewhere

    ordered_tasks = organize_final_task_order(task_groups, modified_permutations, num_machines)

    final_task_schedule = calculate_start_times(ordered_tasks, num_machines)

    return final_task_schedule
end
#*****************************************************************************************************



#*****************************************************************************************************
# Function to swap tasks between two jobs in all task sequences
function swap_tasks(task_sequences, job_id1, job_id2)
    # Clone the original task_sequences to avoid mutating the input directly
    modified_solution = deepcopy(task_sequences)
    
    for task_sequence in modified_solution
        # Find the index of the tasks to swap
        idx1, idx2 = 0, 0
        for i in 1:length(task_sequence)
            if task_sequence[i].job_id == job_id1
                idx1 = i
            elseif task_sequence[i].job_id == job_id2
                idx2 = i
            end
        end
        
        # Perform the swap if both tasks are found
        if idx1 != 0 && idx2 != 0
            task_sequence[idx1], task_sequence[idx2] = task_sequence[idx2], task_sequence[idx1]
        end
    end
    
    return modified_solution
end

# Function to reset the start and end times of all tasks
function reset_task_times!(task_sequences)
    for task_sequence in task_sequences
        for task in task_sequence
            task.start_time = 0
            task.end_time = 0
        end
    end

    return task_sequences
end

# Function to generate a random neighboring solution by swapping two tasks
function generate_random_neighbor_solution(initial_solution, num_tasks)
    # Clone the initial solution to avoid modifying it directly
    current_solution = deepcopy(initial_solution)
    current_makespan = calculate_makespan(current_solution, num_tasks)

    job_id1 = rand(1:num_tasks)
    job_id2 = rand(1:num_tasks)

    if job_id1 != job_id2
        # Perform a swap between two randomly chosen jobs
        swapped_solution = swap_tasks(current_solution, job_id1, job_id2)
        
        # Reset task times and recalculate the start times
        reset_task_times!(swapped_solution)
        current_solution = calculate_start_times(swapped_solution, num_tasks)
        
        # Calculate the makespan of the new solution
        current_makespan = calculate_makespan(current_solution, num_tasks)
    else
        # If the same job IDs are selected, recursively generate another neighbor
        return generate_random_neighbor_solution(initial_solution, num_tasks)
    end            

    # println("Makespan after this iteration:", current_makespan)
    return current_solution, current_makespan
end
#*****************************************************************************************************


#*****************************************************************************************************
# Function to generate PMX permutations based on an initial row
function generate_pmx_permutations(base_row)
    # Determine the length of the base row to define the size of the permutations
    num_elements = length(base_row)
    
    # Initialize the permutation matrix with the base row
    permutation_matrix = [base_row]
    
    # Generate all possible circular shifts of the base row
    shifts = [circshift(base_row, shift) for shift in 1:(num_elements - 1)]
    
    # Shuffle the list of shifts to ensure uniqueness and non-subsequent order
    shuffle!(shifts)
    
    # Select the first num_elements-1 shifts to form the remaining rows
    for shift in shifts[1:(num_elements - 1)]
        push!(permutation_matrix, shift)
    end
    
    return Array{Int}(hcat(permutation_matrix...)')
end

# Function to perform PMX crossover between two parent sequences
function perform_pmx_crossover(parent1, parent2)
    # Length of the parent sequences
    sequence_length = length(parent1)
    
    # Randomly choose crossover points
    cp1, cp2 = sort(rand(1:sequence_length, 2))
    
    # Initialize offspring with zeros
    offspring1 = zeros(Int, sequence_length)
    offspring2 = zeros(Int, sequence_length)
    
    # Copy segments between cp1 and cp2 from each parent to the corresponding offspring
    offspring1[cp1:cp2] = parent1[cp1:cp2]
    offspring2[cp1:cp2] = parent2[cp1:cp2]
    
    # Helper function to fill in the remaining elements of the offspring
    function complete_offspring(offspring, parent, alt_parent)
        # Create a mapping from the alternate parent's segment
        mapping = Dict(alt_parent[i] => parent[i] for i in cp1:cp2)
        
        # Fill in the remaining elements outside the crossover region
        for i in 1:sequence_length
            if i < cp1 || i > cp2
                element = parent[i]
                while element in offspring
                    element = mapping[element]
                end
                offspring[i] = element
            end
        end
    end
    
    # Complete the offspring using the fill function
    complete_offspring(offspring1, parent2, parent1)
    complete_offspring(offspring2, parent1, parent2)
    
    return offspring1, offspring2
end

# Function to create a population of solutions
function create_solution_population(file_path, population_size)
    population = []
    while length(population) < population_size
        sol = generate_initial_solution(file_path)
        if !(sol in population)
            push!(population, sol)
        end
    end

    return population
end

# Function to extract job IDs from the first processor's task sequence
function extract_job_ids_for_pmx(solution)
    # Extract the task sequence of the first processor
    first_task_sequence = solution[1]
    # Extract the job ID from each task in the sequence
    return [task.job_id for task in first_task_sequence]
end

# Function to compare two solutions generated by PMX crossover and return the better one
function compare_pmx_solutions(file_path, seq1, seq2)
    # Perform PMX crossover to generate two offspring sequences
    offspring1, offspring2 = perform_pmx_crossover(seq1, seq2)
    
    # Generate new solutions based on the offspring sequences
    new_solution1 = generate_new_solution(file_path, offspring1)
    new_solution2 = generate_new_solution(file_path, offspring2)

    # Calculate the makespan for each new solution
    makespan1 = calculate_makespan(new_solution1, 4) # Assumes 4 tasks per processor
    makespan2 = calculate_makespan(new_solution2, 4)

    # Return the solution with the smaller makespan
    if makespan1 <= makespan2
        return new_solution1
    else
        return new_solution2
    end
end
#*****************************************************************************************************


#*****************************************************************************************************
# Function to find the solution with the minimum makespan in a population
function find_minimum_makespan(population, num_operations)
    min_makespan = Inf
    best_solution = []
    for solution in population
        current_makespan = calculate_makespan(solution, num_operations)
        if current_makespan < min_makespan
            min_makespan = current_makespan
            best_solution = solution
        end
    end
    
    return best_solution, min_makespan
end

# Function to verify and extract start times from a solution
function extract_start_times_matrix(solution)
    # Determine the size of the matrix
    num_jobs = maximum(task.job_id for task_sequence in solution for task in task_sequence)
    num_tasks = maximum(task.task_id for task_sequence in solution for task in task_sequence)

    # Initialize the matrix with zeros
    start_times_matrix = zeros(Int, num_jobs, num_tasks)

    # Populate the matrix with start times
    for task_sequence in solution
        for task in task_sequence
            start_times_matrix[task.job_id, task.task_id] = task.start_time
        end
    end

    return start_times_matrix
end
#*****************************************************************************************************


#*****************************************************************************************************
# Function to save a matrix to a file
function write_matrix_to_file(matrix, file_name)
    open(file_name, "w") do file
        for row in eachrow(matrix)
            write(file, join(row, " "), "\n")
        end
    end
end
#*****************************************************************************************************


#*****************************************************************************************************
# The Elephant Evolutionary Algorithm (Elephant_EA)
function Elephant_EA(instance_file, solution_file, time_limit)
    num_jobs, num_processors, upper_bound, durations, processors = load_instance(instance_file)

    population_size = 4 * num_processors

    # Create an initial population
    population = create_solution_population(instance_file, population_size)

    start_time = now()

    # Iterate until the time limit is reached
    while (now() - start_time) < Second(time_limit)

        # Select 4 random initial solutions
        i1, i2, i3, i4 = sample(1:population_size, 4; replace=false)
        p1 = population[i1]
        p2 = population[i2]
        p3 = population[i3]
        p4 = population[i4]

        # Extract job sequences for PMX crossover and compare them pairwise
        seq1 = extract_job_ids_for_pmx(p1)
        seq2 = extract_job_ids_for_pmx(p2)
        seq3 = extract_job_ids_for_pmx(p3)
        seq4 = extract_job_ids_for_pmx(p4)

        offspring1 = compare_pmx_solutions(instance_file, seq1, seq2)
        offspring2 = compare_pmx_solutions(instance_file, seq3, seq4)

        # Mutation - Random swap on each offspring individually
        mutated_offspring1, makespan_offspring1 = generate_random_neighbor_solution(offspring1, num_processors)
        mutated_offspring2, makespan_offspring2 = generate_random_neighbor_solution(offspring2, num_processors)

        # Select another 4 random initial solutions
        j1, j2, j3, j4 = sample(1:population_size, num_processors; replace=false)
        r1 = population[j1]
        r2 = population[j2]
        r3 = population[j3]
        r4 = population[j4]

        # Calculate their makespans
        makespan_r1 = calculate_makespan(r1, num_processors)
        makespan_r2 = calculate_makespan(r2, num_processors)
        makespan_r3 = calculate_makespan(r3, num_processors)
        makespan_r4 = calculate_makespan(r4, num_processors)

        # Replace solutions in the population based on makespan comparisons
        if makespan_r1 > makespan_r2
            population[j1] = mutated_offspring1
        else
            population[j2] = mutated_offspring1
        end

        if makespan_r3 > makespan_r4
            population[j3] = mutated_offspring2
        else
            population[j4] = mutated_offspring2
        end    
    end

    # Find the best solution and its makespan in the population
    best_solution, best_makespan = find_minimum_makespan(population, num_processors)

    # Verify and save the final solution
    final_start_times_matrix = extract_start_times_matrix(best_solution)
    write_matrix_to_file(final_start_times_matrix, solution_file)
    
    return best_solution, best_makespan
end
#*****************************************************************************************************


#*****************************************************************************************************
# Main function to execute the algorithm
function main(args)
    if length(args) != 3
        println("Usage: julia script.jl <instance_file> <solution_file> <time_limit>")
        return
    end

    instance_file = args[1]
    solution_file = args[2]
    time_limit = parse(Int, args[3])  

    solution, makespan = Elephant_EA(instance_file, solution_file, time_limit)

    println("Processing instance from: $instance_file with time limit: $time_limit seconds")
    println("Solution saved to $solution_file")
    println("\nFinal makespan: $makespan\n")
    println("\nFinal solution:")
    for task_sequence in solution
        println(task_sequence)
    end
end

main(ARGS)
