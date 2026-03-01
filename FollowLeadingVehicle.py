#!/usr/bin/env python

# Copyright (c) 2018-2020 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

timeout =                                  #Simulation duration
other_vehicle_distance =                   #distance between ego vehicle and lead vehicle
other_vehicle_speed =                      #speed  of lead vehicle
                                           
import py_trees
import carla
import time
import os
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenarios.basic_scenario import BasicScenario
from srunner.tools.scenario_helper import get_waypoint_in_distance
from srunner.scenariomanager.scenarioatomics.atomic_criteria import CollisionTest
from srunner.scenariomanager.scenarioatomics.atomic_behaviors import Idle


class FollowLeadingVehicle(BasicScenario):

    def __init__(self, world, ego_vehicles, config, randomize=False, debug_mode=False, criteria_enable=True, timeout=timeout):
        self._map = CarlaDataProvider.get_map()
        self._lead_start_distance = other_vehicle_distance 
        self._lead_speed = other_vehicle_speed # km/h
 

        self._reference_waypoint = self._map.get_waypoint(config.trigger_points[0].location)
        self.timeout = timeout

        super(FollowLeadingVehicle, self).__init__("FollowLeadVehicle",
                                                   ego_vehicles,
                                                   config,
                                                   world,
                                                   debug_mode,
                                                   criteria_enable=criteria_enable)

    def _initialize_actors(self, config):
        waypoint, _ = get_waypoint_in_distance(self._reference_waypoint, self._lead_start_distance)
        transform = waypoint.transform
        transform.location.z += 0.5
        lead_vehicle = CarlaDataProvider.request_new_actor('vehicle.nissan.patrol', transform)
        self.other_actors.append(lead_vehicle)

    def _create_behavior(self):
        lead = self.other_actors[0]
        ego = self.ego_vehicles[0]

        controller = LeadVehicleController(
            lead_vehicle=lead,
            target_speed_mps=self._lead_speed / 3.6,
            brake_distances=[80], # Start braking after 100 meters
            wait_time=25.0
        )

        infinite_wait = Idle(name="KeepScenarioAlive")
        scenario = py_trees.composites.Sequence("LeadVehicleScenario")
        scenario.add_child(controller)
        scenario.add_child(infinite_wait)

        return scenario

    def _create_test_criteria(self):
        return [CollisionTest(self.ego_vehicles[0])]

    def __del__(self):
        self.remove_all_actors()


class LeadVehicleController(py_trees.behaviour.Behaviour):
    def __init__(self, lead_vehicle, target_speed_mps, brake_distances, wait_time=5.0, name="LeadVehicleController"):
        super(LeadVehicleController, self).__init__(name)
        self.lead_vehicle = lead_vehicle
        self.target_speed = target_speed_mps
        self.brake_distances = sorted(brake_distances)
        self.wait_time = wait_time

        self._initial_location = None
        self._current_stop_index = 0
        self._braking = False
        self._wait_start_time = None
        self._running = False
        self._restarting = False

        self._brake_start_time = None
        self._brake_phase_started = False
        self._reaction_time = 0.7
        self._mu = 0.9
        self._g = 9.81
        self._initial_speed = self.target_speed
        self._required_deceleration = None

        # [Smooth Start]
        self._startup = True
        self._start_time = None
        self._acceleration = 1.5  # m/s²
        self._current_speed = 0.0  # smoothing base
        self._smoothing_factor = 0.1  # between 0 and 1

    def initialise(self):
        self._initial_location = self.lead_vehicle.get_location()
        self._current_stop_index = 0
        self._braking = False
        self._wait_start_time = None
        self._running = True
        self._restarting = False
        self._brake_start_time = None
        self._brake_phase_started = False
        self._initial_speed = self.target_speed
        self._required_deceleration = None

        self._startup = True
        self._start_time = time.time()
        self._current_speed = 0.0

        print(f"[LeadVehicleController] Initialized at {self._initial_location}")

    def update(self):
        if not self._running:
            return py_trees.common.Status.FAILURE

        current_time = time.time()

        # [Smooth Acceleration Phase]
        if self._startup:
            elapsed = current_time - self._start_time
            raw_speed = min(self.target_speed, self._acceleration * elapsed)  #   V=at, linear accealaration,Calculating raw speed when starting

            # [Smooth the transition to avoid jitter]
            self._current_speed += self._smoothing_factor * (raw_speed - self._current_speed)  #  V=Vcurrent _alpha*(Vraw-Vcurrent), Smooth speed ramp-up

            forward = self.lead_vehicle.get_transform().get_forward_vector()
            velocity = carla.Vector3D(forward.x * self._current_speed,
                                      forward.y * self._current_speed,
                                      forward.z * self._current_speed)
            self.lead_vehicle.set_target_velocity(velocity)

            if self._current_speed >= self.target_speed - 0.1:
                self._startup = False
                self._current_speed = self.target_speed
                print(f"[LeadVehicleController] Reached cruising speed smoothly: {self._current_speed:.2f} m/s")

            return py_trees.common.Status.RUNNING

        if self._current_stop_index >= len(self.brake_distances):
            self._set_velocity(self.target_speed)
            return py_trees.common.Status.RUNNING

        current_location = self.lead_vehicle.get_location()    #  Monitor Distance During Driving
        distance_traveled = current_location.distance(self._initial_location)
        next_brake_distance = self.brake_distances[self._current_stop_index]

        if not self._braking and distance_traveled >= next_brake_distance:
            print(f"[LeadVehicleController] Triggering braking at {next_brake_distance} m!")
            self._braking = True
            self._brake_start_time = None
            self._brake_phase_started = False
            self._initial_speed = self.target_speed
            return py_trees.common.Status.RUNNING

        if self._braking and not self._restarting:
            if self._brake_start_time is None:
                self._brake_start_time = current_time

            elapsed = current_time - self._brake_start_time

            if elapsed < self._reaction_time:   #The vehicle doesn’t brake immediately. some delay due to reaction
                self._set_velocity(self.target_speed)   # Keep moving with same speed speed
            else:
                if not self._brake_phase_started:
                    self._brake_phase_started = True
                    braking_distance = (self._initial_speed ** 2) / (2 * self._mu * self._g)  #once the reaction time is over than start slwoing down
                    self._required_deceleration = self._mu * self._g #Max deceleration
                    print(f"[LeadVehicleController] Starting braking... braking_distance={braking_distance:.2f} m, deceleration={self._required_deceleration:.2f} m/s²")

                braking_elapsed = elapsed - self._reaction_time  #how long you've been braking
                current_speed = max(0.0, self._initial_speed - self._required_deceleration * braking_elapsed)  #v=v0−a⋅t uniform deaccearation

                self._set_velocity(current_speed, apply_brake=True)

                if current_speed <= 0.2:
                    print("[LeadVehicleController] Vehicle stopped. Waiting...")
                    self.lead_vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
                    self._wait_start_time = current_time
                    self._restarting = True

            return py_trees.common.Status.RUNNING

        if self._restarting:
            elapsed = current_time - self._wait_start_time
            if elapsed >= self.wait_time:
                print("[LeadVehicleController] Resuming motion.")
                self.lead_vehicle.apply_control(carla.VehicleControl(brake=0.0))
                self._initial_location = self.lead_vehicle.get_location()
                self._braking = False
                self._restarting = False
                self._wait_start_time = None
                self._brake_start_time = None
                self._brake_phase_started = False
                self._initial_speed = self.target_speed
                self._current_stop_index += 1

            return py_trees.common.Status.RUNNING

        self._set_velocity(self.target_speed)
        return py_trees.common.Status.RUNNING

    def _set_velocity(self, speed, apply_brake=False):
        self._current_speed += self._smoothing_factor * (speed - self._current_speed)
        forward = self.lead_vehicle.get_transform().get_forward_vector()
        velocity = carla.Vector3D(forward.x * self._current_speed,
                                  forward.y * self._current_speed,
                                  forward.z * self._current_speed)
        self.lead_vehicle.set_target_velocity(velocity)
        if apply_brake:
            brake_ratio = min(1.0, self._required_deceleration / (self._mu * self._g))
            self.lead_vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=brake_ratio))

    def terminate(self, new_status):
        print("[LeadVehicleController] Terminated.")
        self._running = False



class FollowLeadingVehicleWithObstacle(BasicScenario):

   

    timeout = 90           # Timeout of scenario in seconds

    def __init__(self, world, ego_vehicles, config, randomize=False, debug_mode=False, criteria_enable=True):
        """
        Setup all relevant parameters and create scenario
        """
        self._map = CarlaDataProvider.get_map()
        self._first_actor_location = 25
        self._second_actor_location = self._first_actor_location + 41
        self._first_actor_speed = 10
        self._second_actor_speed = 1.5
        self._reference_waypoint = self._map.get_waypoint(config.trigger_points[0].location)
        self._other_actor_max_brake = 1.0
        self._first_actor_transform = None
        self._second_actor_transform = None

        super(FollowLeadingVehicleWithObstacle, self).__init__("FollowLeadingVehicleWithObstacle",
                                                               ego_vehicles,
                                                               config,
                                                               world,
                                                               debug_mode,
                                                               criteria_enable=criteria_enable)
        if randomize:
            self._ego_other_distance_start = random.randint(4, 8)

    def _initialize_actors(self, config):
        """
        Custom initialization
        """

        first_actor_waypoint, _ = get_waypoint_in_distance(self._reference_waypoint, self._first_actor_location)
        second_actor_waypoint, _ = get_waypoint_in_distance(self._reference_waypoint, self._second_actor_location)
        first_actor_transform = carla.Transform(
            carla.Location(first_actor_waypoint.transform.location.x,
                           first_actor_waypoint.transform.location.y,
                           first_actor_waypoint.transform.location.z - 500),
            first_actor_waypoint.transform.rotation)
        self._first_actor_transform = carla.Transform(
            carla.Location(first_actor_waypoint.transform.location.x,
                           first_actor_waypoint.transform.location.y,
                           first_actor_waypoint.transform.location.z + 1),
            first_actor_waypoint.transform.rotation)
        yaw_1 = second_actor_waypoint.transform.rotation.yaw + 90
        second_actor_transform = carla.Transform(
            carla.Location(second_actor_waypoint.transform.location.x,
                           second_actor_waypoint.transform.location.y,
                           second_actor_waypoint.transform.location.z - 500),
            carla.Rotation(second_actor_waypoint.transform.rotation.pitch, yaw_1,
                           second_actor_waypoint.transform.rotation.roll))
        self._second_actor_transform = carla.Transform(
            carla.Location(second_actor_waypoint.transform.location.x,
                           second_actor_waypoint.transform.location.y,
                           second_actor_waypoint.transform.location.z + 1),
            carla.Rotation(second_actor_waypoint.transform.rotation.pitch, yaw_1,
                           second_actor_waypoint.transform.rotation.roll))

        first_actor = CarlaDataProvider.request_new_actor(
            'vehicle.nissan.patrol', first_actor_transform)
        second_actor = CarlaDataProvider.request_new_actor(
            'vehicle.diamondback.century', second_actor_transform)

        first_actor.set_simulate_physics(enabled=False)
        second_actor.set_simulate_physics(enabled=False)
        self.other_actors.append(first_actor)
        self.other_actors.append(second_actor)

    def _create_behavior(self):
   
        driving_to_next_intersection = py_trees.composites.Parallel(
            "Driving towards Intersection",
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)

        obstacle_clear_road = py_trees.composites.Parallel("Obstalce clearing road",
                                                           policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        obstacle_clear_road.add_child(DriveDistance(self.other_actors[1], 4))
        obstacle_clear_road.add_child(KeepVelocity(self.other_actors[1], self._second_actor_speed))

        stop_near_intersection = py_trees.composites.Parallel(
            "Waiting for end position near Intersection",
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        stop_near_intersection.add_child(WaypointFollower(self.other_actors[0], 10))
        stop_near_intersection.add_child(InTriggerDistanceToNextIntersection(self.other_actors[0], 20))

        driving_to_next_intersection.add_child(WaypointFollower(self.other_actors[0], self._first_actor_speed))
        driving_to_next_intersection.add_child(InTriggerDistanceToVehicle(self.other_actors[1],
                                                                          self.other_actors[0], 15))

        # end condition
        endcondition = py_trees.composites.Parallel("Waiting for end position",
                                                    policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ALL)
        endcondition_part1 = InTriggerDistanceToVehicle(self.other_actors[0],
                                                        self.ego_vehicles[0],
                                                        distance=20,
                                                        name="FinalDistance")
        endcondition_part2 = StandStill(self.ego_vehicles[0], name="FinalSpeed", duration=1)
        endcondition.add_child(endcondition_part1)
        endcondition.add_child(endcondition_part2)

        # Build behavior tree
        sequence = py_trees.composites.Sequence("Sequence Behavior")
        sequence.add_child(ActorTransformSetter(self.other_actors[0], self._first_actor_transform))
        sequence.add_child(ActorTransformSetter(self.other_actors[1], self._second_actor_transform))
        sequence.add_child(driving_to_next_intersection)
        sequence.add_child(StopVehicle(self.other_actors[0], self._other_actor_max_brake))
        sequence.add_child(TimeOut(3))
        sequence.add_child(obstacle_clear_road)
        sequence.add_child(stop_near_intersection)
        sequence.add_child(StopVehicle(self.other_actors[0], self._other_actor_max_brake))
        sequence.add_child(endcondition)
        sequence.add_child(ActorDestroy(self.other_actors[0]))
        sequence.add_child(ActorDestroy(self.other_actors[1]))

        return sequence

    def _create_test_criteria(self):
      
        criteria = []

        collision_criterion = CollisionTest(self.ego_vehicles[0])

        criteria.append(collision_criterion)

        return criteria

    def __del__(self):
        """
        Remove all actors upon deletion
        """
        self.remove_all_actors()


class FollowLeadingVehicleRoute(BasicScenario):

    """
    This class is the route version of FollowLeadingVehicle where the backgrounda activity is used,
    instead of spawning a specific vehicle and making it brake.

    This is a single ego vehicle scenario
    """

    timeout = 120            # Timeout of scenario in seconds

    def __init__(self, world, ego_vehicles, config, randomize=False, debug_mode=False, criteria_enable=True,
                 timeout=60):
        """
        Setup all relevant parameters and create scenario
        """
        self.timeout = timeout
        self._stop_duration = 15
        self._end_time_condition = 30

        super(FollowLeadingVehicleRoute, self).__init__("FollowLeadingVehicleRoute",
                                                        ego_vehicles,
                                                        config,
                                                        world,
                                                        debug_mode,
                                                        criteria_enable=criteria_enable)

    def _initialize_actors(self, config):
        """
        Custom initialization
        """
        pass

    def _create_behavior(self):
        """
        Uses the Background Activity API to force a hard break on the vehicles in front of the actor,
        then waits for a bit to check if the actor has collided.
        """
        sequence = py_trees.composites.Sequence("FollowLeadingVehicleRoute")
        sequence.add_child(Scenario2Manager(self._stop_duration))
        sequence.add_child(Idle(self._end_time_condition))

        return sequence

    def _create_test_criteria(self):
        """
        A list of all test criteria will be created that is later used
        in parallel behavior tree.
        """
        criteria = []
        criteria.append(CollisionTest(self.ego_vehicles[0]))

        return criteria

    def __del__(self):
        """
        Remove all actors upon deletion
        """
        self.remove_all_actors()
