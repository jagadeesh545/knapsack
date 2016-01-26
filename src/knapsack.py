'''
This file contains support code for B551 Hw6
# File version:  November 19, 2015 #

For questions related to genetic algorithms or the knapsack problem, any AI can be of help. For questions related to the support code itself, 
contact Alex at aseewald@indiana.edu.
'''
import math
import copy
import random
import pickle
import itertools
import numpy as np
import pandas as pd
from scipy.stats import norm
from __builtin__ import str

def fitness(max_volume, volumes, prices):
    '''
    This should return a scalar which is to be maximized.
    max_volume is the maximum volume that the knapsack can contain.
    volumes is a list containing the volume of each item in the knapsack.
    prices is a list containing the price of each item in the knapsack, which is aligned with 'volumes'.
    '''
    # calculate total price for chromosome.
    total_volume = sum(volumes)
    total_price = sum(prices)
    chrom_length = len(volumes)   
    while total_volume > max_volume:
        rand = random.randint(0, chrom_length-1)
        for i in range(chrom_length):
            if volumes[rand] > 0:
                volumes[rand] = 0
                prices[rand] = 0
                break
            else:
                rand += 1
                if rand == chrom_length:
                    rand = 0
          
        total_volume = sum(volumes)
        total_price = sum(prices) 
                     
    return total_price

def actualFitness(max_volume, volumes, prices):
    '''
    This should return a scalar which is to be maximized.
    max_volume is the maximum volume that the knapsack can contain.
    volumes is a list containing the volume of each item in the knapsack.
    prices is a list containing the price of each item in the knapsack, which is aligned with 'volumes'.
    '''
    # calculate total price for chromosome.
    total_volume = sum(volumes)
    total_price = sum(prices)
    if total_volume > max_volume:
        return 0                     
    return total_price

def randomSelection(population, fitnesses):
    '''
    This should return a single chromosome from the population. The selection process should be random, but with weighted probabilities 
    proportional to the corresponding 'fitnesses' values
    '''
    sum_of_fitnesses = sum(fitnesses)    
    rnd = random.random() * sum_of_fitnesses
    return_value = []
    for p,f in itertools.izip(population, fitnesses):
        if  rnd < f:
            return_value = p
            break
        rnd -= f
    return return_value

def randomSelectionInverse(population, fitnesses):
    '''
    This should return a index from fitnesses. The index returned is the for minimum fitness value.
    '''
    return fitnesses.index(min(fitnesses))
    
def reproduce(mom, dad):
    "This does genetic algorithm crossover. This takes two chromosomes, mom and dad, and returns two chromosomes."
#     print("mom "+str(mom))
#     print("dad "+str(dad))
    son = list(dad)
    daughter = list(mom)
    
    length = len(dad)
    crossoverPoint = random.randint(1, length-2)
    
    for i in range(crossoverPoint, length):
        son[i] = mom[i]
        daughter[i] = dad[i]
    
    return son, daughter

def mutate(child, probability):
    "Takes a child, produces a mutated child."
    for i in range(len(child)):
        n = random.random()
        if n <= probability:
            if child[i] == 0:
                child[i] = 1
            else:
                child[i] = 0
    return
    
def compute_fitnesses(world, chromosomes):
    '''
    Takes an instance of the knapsack problem and a list of chromosomes and returns the fitness of these chromosomes, according to your 
    'fitness' function. Using this is by no means required, if you want to calculate the fitnesses in your own way, but having a single 
    definition for this is convenient because (at least in my solution) it is necessary to calculate fitnesses at two distinct points 
    in the loop (making a function abstraction desirable).

    Note, 'chromosomes' is required to be a 2D numpy array of boolean values (a fixed-size boolean array is the recommended encoding of 
    a chromosome, and there should be multiple of these arrays, hence the matrix).
    '''
    return [fitness(world[0], world[1] * chromosome, world[2] * chromosome) for chromosome in chromosomes]

def genetic_algorithm(world, popsize, max_years, mutation_probability):
    '''
    world is a data structure describing the problem to be solved, which has a form like 'easy' or 'medium' as defined in the 'run' function.
    The other arguments to this function are what they sound like.
    genetic_algorithm *must* return a list of (chromosomes,fitnesses) tuples, where chromosomes is the current population of chromosomes, 
    and fitnesses is the list of fitnesses of these chromosomes. 
    '''
    length_of_chromosome = len(world[1])
    
    # Generate initial population of chromosomes    
    fitness_value = 0
    result = []

    fitnesses = []
    chromosomes = []
    for j in range(popsize):
        new = []
        for i in range(length_of_chromosome):
            new.append(0)
        chromosomes.append(new)

    for i in range(int(popsize/2)):
        fitness_value = -1
        while fitness_value < 0:
            n = random.getrandbits(length_of_chromosome)
            for j in range(length_of_chromosome):
                chromosomes[i][j] = int((n & (1 << j)) >> j)
#             print("generated chromosome " + str(chromosomes[i]))
            fitness_value = actualFitness(world[0], [a*b for a,b in zip(world[1], chromosomes[i])] , [a*b for a,b in zip(world[2], chromosomes[i])])
            
#         print("evaluated fitness " + str(fitness_value))
        fitnesses.append(fitness_value)
    
    for i in range(int(popsize/2), popsize):
        fitness_value = actualFitness(world[0], [a*b for a,b in zip(world[1], chromosomes[i])] , [a*b for a,b in zip(world[2], chromosomes[i])])
        fitnesses.append(fitness_value)
    
    result.append((chromosomes, fitnesses))
    
    flag = False
    for k in range(max_years):
        for i in range(popsize):
            fitness_value = fitness(world[0], [a*b for a,b in zip(world[1], chromosomes[i])] , [a*b for a,b in zip(world[2], chromosomes[i])])
            fitnesses[i] = fitness_value
        
        # break when 90% of the fitnesses are same
        fitness_values = list(set(fitnesses))
        for i in fitness_values:
            if fitnesses.count(i) >= (0.9 * len(fitnesses)):
                flag = True
        if flag == True:
            break
        
        # this loop ensures that we generate atmost 85% of new chromosomes
        for l in range(int((0.85 * popsize)/2)-1):
            # select two random members
            mom = randomSelection(chromosomes, fitnesses)
            dad = randomSelection(chromosomes, fitnesses)
            
            # get their children 
            son, daughter = reproduce(mom, dad)
            
#             print("children before mutation " + str(son) + " " + str(daughter))
            # mutate children
            mutate(son, mutation_probability)
            mutate(daughter, mutation_probability)
#             print("children after mutation " + str(son) + " " + str(daughter))
            
            # add to new population
            chromosomes.append(son)
            fitnesses.append(fitness(world[0], [a*b for a,b in zip(world[1], son)] , [a*b for a,b in zip(world[2], son)]))
            chromosomes.append(daughter)
            fitnesses.append(fitness(world[0], [a*b for a,b in zip(world[1], daughter)] , [a*b for a,b in zip(world[2], daughter)]))
        
        # this loop ensures that we remove weakest chromosomes
        for l in range(int((0.85 * popsize)/2)-1):
            # get two random indices with least fitness
            # to get two least fit members we perform randomSelection with the fitnesses reversed
            try:
                sub1 = randomSelectionInverse(chromosomes, fitnesses)
#                 sub1 = random.randint(0,len(chromosomes)-1)
                del chromosomes[sub1]
                del fitnesses[sub1]
                sub2 = randomSelectionInverse(chromosomes, fitnesses)
#                 sub2 = random.randint(0,len(chromosomes)-1)
                del chromosomes[sub2]
                del fitnesses[sub2]
            except:
                print("lenght of choromsomes list = " + str(len(chromosomes)))
                print("lenght of fitnesses list = " + str(len(fitnesses)))
                print("sub1 = " + str(sub1))
                print("sub2 = " + str(sub2))
                raise
            
#         for c,f in itertools.izip(chromosomes,fitnesses):
#             print("chromosome " + str(c))
#             print("fitness " + str(f))
        
        # append new generation
        for i in range(popsize):
            fitness_value = actualFitness(world[0], [a*b for a,b in zip(world[1], chromosomes[i])] , [a*b for a,b in zip(world[2], chromosomes[i])])
            fitnesses[i] = fitness_value
    
        result.append((chromosomes, fitnesses))
        
#     print(result)
    return result
        
def run(popsize, max_years, mutation_probability):
    '''
    The arguments to this function are what they sound like.
    Runs genetic_algorithm on various knapsack problem instances and keeps track of tabular information with this schema:
    DIFFICULTY YEAR HIGH_SCORE AVERAGE_SCORE BEST_PLAN
    '''
    table = pd.DataFrame(columns=["DIFFICULTY", "YEAR", "HIGH_SCORE", "AVERAGE_SCORE", "BEST_PLAN"])
    sanity_check = (10, [10, 5, 8], [100, 50, 80])
    chromosomes = genetic_algorithm(sanity_check, popsize, max_years, mutation_probability)
    print("Years ran for sanity_check = " + str(len(chromosomes)))
    for year, data in enumerate(chromosomes):    
        year_chromosomes, fitnesses = data
        table = table.append({'DIFFICULTY' : 'sanity_check', 'YEAR' : year, 'HIGH_SCORE' : max(fitnesses),
            'AVERAGE_SCORE' : np.mean(fitnesses), 'BEST_PLAN' : year_chromosomes[np.argmax(fitnesses)]}, ignore_index=True)
    easy = (20, [20, 5, 15, 8, 13], [10, 4, 11, 2, 9])
    chromosomes = genetic_algorithm(easy, popsize, max_years, mutation_probability)
    print("Years ran for easy = " + str(len(chromosomes)))
    for year, data in enumerate(chromosomes):
        year_chromosomes, fitnesses = data
        table = table.append({'DIFFICULTY' : 'easy', 'YEAR' : year, 'HIGH_SCORE' : max(fitnesses),
            'AVERAGE_SCORE' : np.mean(fitnesses), 'BEST_PLAN' : year_chromosomes[np.argmax(fitnesses)]}, ignore_index=True)
    medium = (100, [13, 19, 34, 1, 20, 4, 8, 24, 7, 18, 1, 31, 10, 23, 9, 27, 50, 6, 36, 9, 15],
                   [26, 7, 34, 8, 29, 3, 11, 33, 7, 23, 8, 25, 13, 5, 16, 35, 50, 9, 30, 13, 14])
    chromosomes = genetic_algorithm(medium, popsize, max_years, mutation_probability)
    print("Years ran for medium = " + str(len(chromosomes)))
    for year, data in enumerate(chromosomes):
        year_chromosomes, fitnesses = data
        table = table.append({'DIFFICULTY' : 'medium', 'YEAR' : year, 'HIGH_SCORE' : max(fitnesses),
            'AVERAGE_SCORE' : np.mean(fitnesses), 'BEST_PLAN' : year_chromosomes[np.argmax(fitnesses)]}, ignore_index=True)
    hard = (5000, norm.rvs(50, 15, size=100), norm.rvs(200, 60, size=100))
    chromosomes = genetic_algorithm(hard, popsize, max_years, mutation_probability)
    print("Years ran for hard = " + str(len(chromosomes)))
    for year, data in enumerate(chromosomes):
        year_chromosomes, fitnesses = data
        table = table.append({'DIFFICULTY' : 'hard', 'YEAR' : year, 'HIGH_SCORE' : max(fitnesses),
            'AVERAGE_SCORE' : np.mean(fitnesses), 'BEST_PLAN' : year_chromosomes[np.argmax(fitnesses)]}, ignore_index=True)
      
    for difficulty_group in ['sanity_check', 'easy', 'medium', 'hard']:
        group = table[table['DIFFICULTY'] == difficulty_group]
        bestrow = group.ix[group['HIGH_SCORE'].argmax()]
        print("Best year for difficulty {} is {} with high score {} and chromosome {}".format(\
            difficulty_group, int(bestrow['YEAR']), bestrow['HIGH_SCORE'], bestrow['BEST_PLAN']))
    # saves the performance data, in case you want to refer to it later. pickled python objects can be loaded back at any later point.
    table.to_pickle("results.pkl")

population_values = [10, 20, 25, 30, 40, 50, 70, 100]
max_years_values = [5, 10, 20, 30, 40, 50, 70, 100]
mutation_prob_values = [0.1, 0.05, 0.02, 0.01]
for i in population_values:
    for j in max_years_values:
        for k in mutation_prob_values:
            print("Run Config: (" + str(i) + ", " + str(j) + ", " + str(k) + ")")
            for l in range(5):
                run(i, j, k)
            print("-------------------------------------------------------------------------------------------------------------------")
            
