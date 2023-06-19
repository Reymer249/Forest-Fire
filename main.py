import math
import numpy as np
import typing


class DroneExtinguisher:
    def __init__(self, forest_location: typing.Tuple[float, float], bags: typing.List[int],
                 bag_locations: typing.List[typing.Tuple[float, float]],
                 liter_cost_per_km: float, liter_budget_per_day: int, usage_cost: np.ndarray):
        """
        The DroneExtinguisher object. This object contains all functions necessary to compute the most optimal way of saving the forest
        from the fire using dynamic programming. Note that all costs that we use in this object will be measured in liters. 

        :param forest_location: the location (x,y) of the forest 
        :param bags: list of the contents of the water bags in liters
        :param bag_locations: list of the locations of the water bags
        :param liter_cost_per_km: the cost of traveling a kilometer with drones, measured in liters of waters 
        :param liter_budget_per_day: the maximum amount of work (in liters) that we can do per day 
                                     (sum of liter contents transported on the day + travel cost in liters)
        :param usage_cost: a 2D array. usage_cost[i,k] is the cost of flying water bag i with drone k from the water bag location to the forest
        """

        self.forest_location = forest_location
        self.bags = bags
        self.bag_locations = bag_locations
        self.liter_cost_per_km = liter_cost_per_km
        self.liter_budget_per_day = liter_budget_per_day
        self.usage_cost = usage_cost  # usage_cost[i,k] = additional cost to use drone k to for bag i

        # the number of bags and drones that we have in the problem
        self.num_bags = len(self.bags)
        self.num_drones = self.usage_cost.shape[1] if not usage_cost is None else 1

        # list of the travel costs measured in the amount of liters of water
        # that could have been emptied in the forest (measured in integers)
        self.travel_costs_in_liters = []

        # idle_cost[i,j] is the amount of time measured in liters that we are idle on a day if we 
        # decide to empty bags[i:j+1] on that day
        self.idle_cost = -1 * np.ones((self.num_bags, self.num_bags))

        # optimal_cost[i,k] is the optimal cost of emptying water bags[:i] with drones[:k+1]
        # this has to be filled in using the dynamic programming function
        self.optimal_cost = np.zeros((self.num_bags + 1, self.num_drones))

        # Data structure that can be used for the backtracing method (NOT backtracking):
        # reconstructing what bags we empty on every day in the forest
        self.backtrace_memory = dict()

        # costs_dp_array[i][j] represents a tuple, in which 1st number is a minimal cost
        # (including last day idle loss) of delivering bags[:i] with drones[:j], ending
        # up with the drone j. The 2nd number is hte cost without the last day idle loss,
        # so the one we will need for the final answer. And the 3rd tuple is a coordinates
        # of the last bag on the previous day. If y = -1, then we are now on the last bag
        # delivered on the last day.
        self.costs_dp_array = np.empty((self.num_bags, self.num_drones), dtype=object)
        self.costs_dp_array.fill((0, 0, (0, 0)))


    @staticmethod
    def compute_euclidean_distance(point1: typing.Tuple[float, float], point2: typing.Tuple[float, float]) -> float:
        """
        A static method (as it does not have access to the self. object) that computes the Euclidean
        distance between two points

        :param point1: an (x,y) tuple indicating the location of point 1
        :param point2: idem for point2

        Returns 
          float: the Euclidean distance between the two points
        """

        # we use Euclidean distance formula for the return
        return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

    def fill_travel_costs_in_liters(self):
        """
        Function that fills in the self.travel_costs_in_liters data structure such that
        self.travel_costs_in_liters[i] is the cost of traveling from the forest/drone housing
        to the bag AND back to the forest, measured in liters of waters (using liter_cost_per_km)
        Note: the cost in liters should be rounded up (with, e.g., np.ceil)
                
        The function does not return anything.  
        """

        # for every bag's location we compute the distance
        # as the bags are sorted, we may simply use append method
        for bag_location in self.bag_locations:
            self.travel_costs_in_liters.append(
                np.ceil(self.compute_euclidean_distance(bag_location, self.forest_location)
                        * self.liter_cost_per_km
                        * 2))

    def compute_sequence_idle_time_in_liters(self, i, j):
        """
        Function that computes the idle time (time not spent traveling to/from bags or emptying bags in the forest)
        in terms of liters. This function assumes that self.travel_costs_in_liters has already been filled with the
        correct values using the function above, as it makes use of that data structure.
        More specifically, this function computes the idle time on a day if we decide to empty self.bags[i:j+1] 
        (bag i, bag i+1, ..., bag j) on that day.

        Note: the returned idle time can be negative (if transporting the bags is not possible within a day) 

        :param i: integer index 
        :param j: integer index

        Returns:
          int: the amount of time (measured in liters) that we are idle on the day   
        """

        # the general amount of liters we spent during the day
        work_litres = 0

        # go through every bag in a given section and append their expenses to the work_litres
        for k in range(i, j + 1):
            work_litres += self.travel_costs_in_liters[k] + self.bags[k]

        # return the difference between the budget we have and the work_litres
        return self.liter_budget_per_day - work_litres

    def compute_idle_cost(self, i, j, idle_time_in_liters):
        """
        Function that transforms the amount of time that we are idle on a day if we empty self.bags[i:j+1]
        on a day (idle_time_in_liters) into a quantity that we want to directly optimize using the formula
        in the assignment description. 
        If transporting self.bags[i:j+1] is not possible within a day, we should return np.inf as cost. 
        Moreover, if self.bags[i:j+1] are the last bags that are transported on the final day, the idle cost is 0 
        as the operation has been completed. In all other cases, we use the formula from the assignment text. 

        You may not need to use every argument of this function

        :param i: integer index
        :param j: integer index
        :param idle_time_in_liters: the amount of time that we are idle on a day measured in liters

        Returns
          - integer: the cost of being idle on a day corresponding to idle_time_in_liters
        """

        if idle_time_in_liters < 0:  # this constraint is checked first as the 1st priority
            return np.inf  # because even if we are on the last day, but working liters are exceeding possible liters
            # we have to output infinity, not 0
        elif j == self.num_bags - 1:  # check for the last day
            return 0
        else:
            return idle_time_in_liters ** 3

    def compute_sequence_usage_cost(self, i: int, j: int, k: int) -> float:
        """
        Function that computes and returns the cost of using drone k for self.bags[i:j+1], making use of
        self.usage_cost, which gives the cost for every bag-drone pair. 
        Note: the usage cost is independent of the distance to the forest. This is purely the operational cost
        to use drone k for bags[i:j+1].

        :param i: integer index
        :param j: integer index
        :param k: integer index

        Returns
          - float: the cost of usign drone k for bags[i:j+1] 
        """

        cost = 0  # the final cost to be output

        # going through all bags and adding their transporting cost using drone k to the cost
        for p in range(i, j+1):
            cost += self.usage_cost[p, k]

        return cost

    def fill_full_travel_cost(self):
        """
        Function that fills in the self.full_travel_cost data structure

        In a way, that self.full_travel_cost[i] is the cost of traveling from the forest/drone housing
        to the bag AND back to the forest, AND emptying the bag, measured in liters of waters.
        We introduce this data structure for the convenience of the computations.

        The function does not return anything.
        """

        self.full_travel_cost = self.travel_costs_in_liters.copy()

        # going through all the costs and adding the liters in the bag
        for i in range(len(self.full_travel_cost)):
            self.full_travel_cost[i] += self.bags[i]

    def dynamic_programming(self):
        """
        The function that uses dynamic programming to solve the problem: compute the optimal way of emptying bags in the forest
        per day and store a solution that can be used in the backtracing function below (if you want to do that assignment part). 
        In this function, we fill the memory structures self.idle_cost and self.optimal_cost making use of functions defined above. 
        This function does not return anything. 
        """

        # executing this function as we will need the list with the full cost of work (self.full_travel_cost)
        self.fill_full_travel_cost()

        # This function will take self.costs_dp_array data structure and the numbers:
        # i - row of the array, j_1 - 1st column, j_2 - second column. It finds
        # the tuple with the minimal 1st value (cost including last day idle loss) in
        # the given row i and from column j_1 to the column j_2 included.
        # The output is the tuple of the found tuple and its position
        # in the self.costs_dp_array.
        def find_min_tuple(array, i, j_1, j_2):

            # min_tuple is tuple with the minimal value which we will output
            min_tuple = (np.inf, 0, (0, 0))
            # min col - column in which min_tuple is located
            min_col = 0

            # going through every column inbetween j_1 and j_2
            for k in range(j_1, j_2+1):
                if array[i][k][0] < min_tuple[0]:  # looking for the min value
                    min_tuple = array[i][k]
                    min_col = k

            return (min_tuple, (i, min_col))

        # From now on and up to the end of this function we are filling the self.cost_dp_array data structure

        # This loop fill 1st row of the self.cost_dp_array data structure as it's obvious
        for i in range(len(self.costs_dp_array[0])):
            self.costs_dp_array[0][i] = ((self.liter_budget_per_day - self.full_travel_cost[0])**3 + self.usage_cost[0][i],
                                          self.usage_cost[0][i],
                                          (-1, i))

        # This is the main loop to fill all the rest of the rows
        for row in range(1, len(self.full_travel_cost)):

            # Here we calculate how much previous rows we have to consider in
            # our d.p. equation. collected value is the number of liters we are
            # working the last day, row0 is the index of row up to which
            # we have to consider previous instances.
            collected = self.full_travel_cost[row]
            row0 = row
            while collected <= self.liter_budget_per_day and row0 >= 0:
                row0 -= 1
                collected += self.full_travel_cost[row0]
            row0 += 1

            # Going through every value in a row to fill it up
            for column in range(len(self.costs_dp_array[1])):

                # For every instance we are creating lists with the possible
                # costs to find the minimum and with the coordinates to do the backtracing

                # possible_costs_with_last[i] is the cost including the last day idle loss,
                # if we are using packages[row-i:row] on the last day
                possible_costs_with_last = np.array([0]*(row-row0+1))
                # possible_cost[i] is the cost without the last day idle loss,
                # if we are using packages[row-i:row] on the last day
                possible_costs = np.array([0]*(row-row0+1))
                # possible_col[i] is the coordinate of the package we brought last
                # on the previous day
                possible_col = [(0, 0)]*(row-row0+1)

                # In this loop we are going through all possible d.p. instances
                for m in range(len(possible_costs)):

                    # The cost of travelling on the last day
                    fin_day_travel = 0
                    # The cost of usage on the last day
                    fin_day_usage = 0

                    # This loop is to fill fin_day_travel and fin_day_usage data structures
                    for l in range(m+1):
                        fin_day_travel += self.full_travel_cost[row-l]
                        fin_day_usage += self.usage_cost[row-l][column]

                    # The tuple and its position of the last bag used the precious day
                    # with the minimal cost
                    previous_tuple = find_min_tuple(self.costs_dp_array, row-m-1, 0, column)

                    # Filling the values we found
                    possible_costs_with_last[m] = (self.liter_budget_per_day-fin_day_travel)**3 + fin_day_usage + previous_tuple[0][0]
                    possible_costs[m] = fin_day_usage + previous_tuple[0][0]
                    possible_col[m] = previous_tuple[1]

                # Here we are looking fo the minimal cost we may obtain.
                # If we are not on the last day, we pick the tuple with the minimal
                # cost with the last day idle time. Otherwise, we pick the tuple with
                # the minimal cost as it stated in the text (without the last day)
                if row == len(self.full_travel_cost) - 1:
                    arg = np.argmin(possible_costs)
                else:
                    arg = np.argmin(possible_costs_with_last)

                # Recording the best option
                self.costs_dp_array[row][column] = (possible_costs_with_last[arg], possible_costs[arg], possible_col[arg])

    def lowest_cost(self) -> float:
        """
        Returns the lowest cost at which we can empty the water bags to extinguish to forest fire. Inside of this function,
        you can assume that self.dynamic_progrmaming() has been called so that in this function, you can simply extract and return
        the answer from the filled in memory structure.

        Returns:
          - float: the lowest cost
        """

        # The tuple with the min value
        min_cost_tuple = (0, np.inf, (0, 0))

        # We are going through the last row to find the minimal
        # value of the cost (that's because we are not obligated
        # to finish on the last drone, we may finish on the 1st one
        # or 2nd as well).
        for k in range(self.costs_dp_array.shape[1]):
            if self.costs_dp_array[-1][k][1] < min_cost_tuple[1]:
                min_cost_tuple = self.costs_dp_array[-1][k]

        return min_cost_tuple[1]

    def backtrace_solution(self):
        """
        Returns the solution of how the lowest cost was obtained by using, for example, self.backtrace_memory (but feel free to do it your own way). 
        The solution is a tuple (leftmost indices, drone list) as described in the assignment text. Here, leftmost indices is a list 
        [idx(1), idx(2), ..., idx(T)] where idx(i) is the index of the water bag that is emptied left-most (at the start of the day) on day i. 
        Drone list is a list [d(0), d(1), ..., d(num_bags-1)] where d(j) tells us which drone was used in the optimal
        solution to transport water bag j.  
        See the assignment description for an example solution. 

        This function does not have to be made - you can still pass the assignment if you do not hand this in,
        however it will cost a full point if you do not do this (and the corresponding question in the report).  
            
        :return: A tuple (leftmost indices, drone list) as described above
        """

        # The tuple with the min value
        min_cost_tuple = (0, np.inf, (0, 0))
        col = 0

        # We are going through the last row to find the minimal
        # value of the cost (that's because we are not obligated
        # to finish on the last drone, we may finish on the 1st one
        # or 2nd as well).
        for k in range(self.costs_dp_array.shape[1]):
            if self.costs_dp_array[-1][k][1] < min_cost_tuple[1]:
                min_cost_tuple = self.costs_dp_array[-1][k]
                col = k

        # schedule is the sequence of tuples, in which 1st number represents the
        # final bag we are carrying on some day i, and the 2nd number is the drone
        # we are using that day
        schedule = []
        index = min_cost_tuple[2]

        # Here we are starting from out minimail tuple and go back. As in every
        # tuple we have a coordinates of a previous bag, we simply follow that.
        # However, when the y coordinate becomes -1, it means, that we are on the
        # last bag of the first day, so we may end our trip.
        while index[0] != -1:
            schedule.append(min_cost_tuple[2])
            min_cost_tuple = self.costs_dp_array[index[0]][index[1]]
            index = min_cost_tuple[2]

        # Reversing the schedule as we started at the end and went backwards
        schedule.reverse()
        # Adding the last bag on the last day
        schedule.append((self.costs_dp_array.shape[0]-1, col))

        # In the following code we are transforming schedule into the
        # view needed for the answer

        # List of the indices of the 1st bags on every day
        left_indices = []
        # drone_list[i] represents a drone we are using for the bag i
        drone_list = []

        # Going through every day in schedule and transforming it into the needed format
        for i in range(len(schedule)):
            # As in schedule we are recording the last bag on every day,
            # the first day has to be treated separately
            if i == 0:
                # We append 0 as on the 1st day we surely start from 0 (1st) bag
                left_indices.append(0)
                # After that we append drone we are using on the 1st day n times, where
                # n in equals to the number in the schedule (number of the last bag)
                # +1, as we start counting from 0
                for j in range(schedule[i][0]+1):
                    drone_list.append(schedule[i][1])
            else:
                # We append the bag we finished last day plus 1
                left_indices.append(schedule[i-1][0]+1)
                # We append drone we are using on a day i n times, where n
                # is the difference between the numbers of the last drone
                # we use on the i-th day and on the day before that.
                for j in range(schedule[i-1][0], schedule[i][0]):
                    drone_list.append(schedule[i][1])

        return (left_indices, drone_list)


