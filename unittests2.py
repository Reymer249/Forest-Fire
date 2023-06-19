import typing
import unittest
import numpy as np

from dynprog import DroneExtinguisher


class TestDroneExtinguisher(unittest.TestCase):

    def test_backtrace_solution(self):
        forest_location = (1, 1)
        bags = [1, 1, 1, 1, 1]
        bag_locations = [(1, 0), (0, 1), (2, 1), (1, 2), (1, 1)]
        liter_cost_per_km = 2
        liter_budget_per_day = 15
        usage_cost = np.array([[0, 10000000, 0, 10000000],
                               [10000000, 0, 10000000, 0],
                               [0, 10000000, 0, 10000000],
                               [10000000, 0, 10000000, 0],
                               [0, 10000000, 0, 10000000]])

        backtrace_solution = ([0, 3, 4], [0, 0, 0, 1, 2])

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

        self.assertEqual(backtrace_solution, de.backtrace_solution())

    def test_dynamic_programming(self):
        forest_location = (0, 0)
        bags = [10, 10, 6, 6, 1]
        bag_locations = [(1, 0), (0, 1), (3, 4), (4, 3), (6, 8)]
        liter_cost_per_km = 0.5
        liter_budget_per_day = 100
        usage_cost = np.array([[0, 10000000],
                               [10000000, 0],
                               [0, 10000000],
                               [10000000, 0],
                               [0, 10000000]])

        costs_dp_array_solution = [[(704969.0, 0, (-1, 0)), (10704969.0, 10000000, (-1, 1))],
                                   [(10474552, 10000000, (-1, 0)), (1409938, 704969, (0, 0))],
                                   [(10300763, 10000000, (-1, 0)), (11179521, 10704969, (0, 0))],
                                   [(20175616, 20000000, (-1, 0)), (11005732, 10300763, (2, 0))],
                                   [(20091125, 20000000, (-1, 0)), (20775315, 20300763, (2, 0))]]

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

        self.assertEqual(de.costs_dp_array.tolist(), costs_dp_array_solution)

    def test_dyanmic_programming_with_travel_cost1(self):
        forest_location = (2, 2)
        bags = [1, 2, 1, 2]
        bag_locations = [(0, 0), (4, 4), (3, 1), (2, 2)]
        liter_cost_per_km = 0.1
        liter_budget_per_day = 20
        usage_cost = np.array([[0, 1000, 0, 0],
                               [99999, 1000, 99999, 0],
                               [0, 99999, 0, 99999],
                               [0, 0, 99999, 0]])

        solution = 11207

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

        self.assertEqual(de.lowest_cost(), solution)

    def test_dyanmic_programming_with_travel_cost2(self):
        forest_location = (0, 0)
        bags = [1, 1]
        bag_locations = [(1, 0), (0, 1)]
        liter_cost_per_km = 0.5
        liter_budget_per_day = 4
        usage_cost = np.array([[0, 999],
                               [999, 0]])

        solution = 8

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

        self.assertEqual(de.lowest_cost(), solution)
