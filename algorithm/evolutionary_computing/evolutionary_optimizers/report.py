def report_island_status(test_island):
    print("-----  Generation %d  -----" % test_island.generational_age)
    print("Best individual:     ", test_island.get_best_individual())
    print("Best fitness:        ", test_island.get_best_fitness())
    print("Fitness evaluations: ", test_island.get_fitness_evaluation_count())