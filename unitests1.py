import typing
import unittest
import numpy as np

from dynprog import DroneExtinguisher


class TestDroneExtinguisher(unittest.TestCase):

    def test_euclidean_distance(self):
        point1 = (0,0)
        point2 = (3,4)
        dist = 5

        self.assertEqual(DroneExtinguisher.compute_euclidean_distance(point1, point2), dist)
        self.assertEqual(DroneExtinguisher.compute_euclidean_distance(point2, point1), dist)


    def test_fill_travel_costs_in_liters(self):
        de = DroneExtinguisher(forest_location=(0,0), bags=[10, 30], 
                               bag_locations=[(3,4),(6,8)], liter_cost_per_km=2,
                               liter_budget_per_day=0, usage_cost=None)
        de.fill_travel_costs_in_liters()

        distances = [5, 10]
        liter_costs = [20, 40] # 2x the distances in this case
        self.assertListEqual(liter_costs, de.travel_costs_in_liters)

    def test_fill_travel_costs_in_liters_with_rounding(self):
        de = DroneExtinguisher(forest_location=(0,0), bags=[10, 30], 
                               bag_locations=[(2.3,1),(7,2.7)], liter_cost_per_km=2,
                               liter_budget_per_day=0, usage_cost=None)
        de.fill_travel_costs_in_liters()

        liter_costs = [11, 31] # 2x the distances in this case
        self.assertListEqual(liter_costs, de.travel_costs_in_liters)

    def test_compute_sequence_idle_time_in_liters(self):
        de = DroneExtinguisher(forest_location=(0,0), bags=[10, 30, 1000], 
                               bag_locations=[(2.3,1),(7,2.7), (1,1)], liter_cost_per_km=2,
                               liter_budget_per_day=1000, usage_cost=None)
        de.fill_travel_costs_in_liters()
        
        liter_costs = [11, 31] # 2x the distances in this case
        idle_time_in_liters = [1000-11-10, 1000-21-31-30, -88]
        i = 0 
        for j in range(3):
            self.assertEqual(de.compute_sequence_idle_time_in_liters(i,j), idle_time_in_liters[j])


    def test_compute_idle_cost(self):
        de = DroneExtinguisher(forest_location=(0,0), bags=[10, 30, 1000], 
                               bag_locations=[(2.3,1),(7,2.7), (1,1)], liter_cost_per_km=2,
                               liter_budget_per_day=1000, usage_cost=None)
        de.fill_travel_costs_in_liters()
        
        liter_costs = [11, 31] # 2x the distances in this case
        idle_time_in_liters = [1000-11-10, 1000-21-31-30, -88]
        idle_costs = [938313739, 773620632, np.inf]
        i = 0 
        for j in range(3):
            self.assertEqual(de.compute_idle_cost(i,j, idle_time_in_liters[j]), idle_costs[j])

       
        # no idle cost for the final day
        de = DroneExtinguisher(forest_location=(0,0), bags=[10, 30, 1000], 
                               bag_locations=[(2.3,1),(7,2.7), (1,1)], liter_cost_per_km=2,
                               liter_budget_per_day=1000, usage_cost=None)
        de.fill_travel_costs_in_liters()
        
        liter_costs = [11, 31] # 2x the distances in this case
        idle_time_in_liters = [1000-11-10, 1000-21-31-30, -88]
        idle_costs = [938313739, 773620632, np.inf]
        for i in range(3):
            self.assertEqual(de.compute_idle_cost(i,2,10), 0)

    def test_compute_sequence_usage_cost(self):
        usage_cost = np.array([[10,20,30],
                              [5,25,16]])
        
        de = DroneExtinguisher(forest_location=(0,0), bags=[10, 30, 1000], 
                               bag_locations=[(2.3,1),(7,2.7), (1,1)], liter_cost_per_km=2,
                               liter_budget_per_day=1000, usage_cost=usage_cost)
        

        an1 = 15
        an2 = 45
        an3 = 46

        self.assertEqual(de.compute_sequence_usage_cost(0,1,0), an1)
        self.assertEqual(de.compute_sequence_usage_cost(0,1,1), an2)
        self.assertEqual(de.compute_sequence_usage_cost(0,1,2), an3)


    def test_dynamic_programming_simple(self):
        forest_location = (0,0)
        bags = [6,3,8,9]
        bag_locations = [(0,0) for _ in range(len(bags))] # no travel distance
        liter_cost_per_km = 1 # doesn't matter as there is no travel distance
        liter_budget_per_day = 10
        usage_cost = np.array([[0],
                            [0],
                            [0],
                            [0]])

        solution = 9

        de = DroneExtinguisher(
            forest_location=forest_location,
            bags=bags,
            bag_locations=bag_locations,
            liter_cost_per_km=liter_cost_per_km,
            liter_budget_per_day=liter_budget_per_day,
            usage_cost=usage_cost
        )

        de.fill_travel_costs_in_liters()
        de.dynamic_programming()
        lowest_cost = de.lowest_cost()
        self.assertEqual(lowest_cost, solution)

    def test_dyanmic_programming_one_day(self):
        forest_location = (0,0)
        bags = [10,3,1]
        bag_locations = [(0,0) for _ in range(len(bags))] # no travel distance
        liter_cost_per_km = 1 # doesn't matter as there is no travel distance
        liter_budget_per_day = 20
        usage_cost = np.array([[0],
                            [0],
                            [0],
                            [0]])

        solution = 0

        de = DroneExtinguisher(
            forest_location=forest_location,
            bags=bags,
            bag_locations=bag_locations,
            liter_cost_per_km=liter_cost_per_km,
            liter_budget_per_day=liter_budget_per_day,
            usage_cost=usage_cost
        )

        de.fill_travel_costs_in_liters()
        de.dynamic_programming()
        lowest_cost = de.lowest_cost()
        self.assertEqual(lowest_cost, solution)

    def test_dyanmic_programming_no_travel_cost(self):
        forest_location = (0,0)
        bags = [4,10,3,4,20]
        bag_locations = [(0,0) for _ in range(len(bags))] # no travel distance
        liter_cost_per_km = 1 # doesn't matter as there is no travel distance
        liter_budget_per_day = 20
        usage_cost = np.array([[0],
                            [0],
                            [0],
                            [0],
                            [0]])

        solution = 2413

        de = DroneExtinguisher(
            forest_location=forest_location,
            bags=bags,
            bag_locations=bag_locations,
            liter_cost_per_km=liter_cost_per_km,
            liter_budget_per_day=liter_budget_per_day,
            usage_cost=usage_cost
        )

        de.fill_travel_costs_in_liters()
        de.dynamic_programming()
        lowest_cost = de.lowest_cost()
        self.assertEqual(lowest_cost, solution)


    def test_dyanmic_programming_with_travel_cost(self):
        forest_location = (0,0)
        bags = [3,9,2,3,19]
        bag_locations = [(3,4) for _ in range(len(bags))] # constant travel distance 5
        liter_cost_per_km = 0.1 # now there is a constant cost of 1 liter traveling time per bag
        liter_budget_per_day = 20
        usage_cost = np.array([[1],
                             [1],
                             [1],
                             [1],
                             [1]])

        solution = 2418

        de = DroneExtinguisher(
            forest_location=forest_location,
            bags=bags,
            bag_locations=bag_locations,
            liter_cost_per_km=liter_cost_per_km,
            liter_budget_per_day=liter_budget_per_day,
            usage_cost=usage_cost
        )

        de.fill_travel_costs_in_liters()
        de.dynamic_programming()
        lowest_cost = de.lowest_cost()
        self.assertEqual(lowest_cost, solution)


    def test_dyanmic_programming_with_travel_cost_multiple_drones(self):
        forest_location = (0,0)
        bags = [3,9,2,3,19]
        bag_locations = [(3,4) for _ in range(len(bags))] # constant travel distance 5
        liter_cost_per_km = 0.1 # now there is a constant cost of 1 liter traveling time per bag
        liter_budget_per_day = 20
        usage_cost = np.array([[1,1,0],
                             [1,1,0],
                             [1,1,0],
                             [1,1,0],
                             [1,1,0]])

        solution = 2413

        de = DroneExtinguisher(
            forest_location=forest_location,
            bags=bags,
            bag_locations=bag_locations,
            liter_cost_per_km=liter_cost_per_km,
            liter_budget_per_day=liter_budget_per_day,
            usage_cost=usage_cost
        )

        de.fill_travel_costs_in_liters()
        de.dynamic_programming()
        lowest_cost = de.lowest_cost()
        self.assertEqual(lowest_cost, solution)

