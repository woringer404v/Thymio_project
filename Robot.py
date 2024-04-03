from tdmclient import aw

'''
Wrapper class of the thymio robot, so that we can be agnostic of how it works and of the
asynchronous nature of Aseba, which we do not have to think about.
'''
class Thymio(object):
    '''
    scaling_factor: constant for dimensions changes, has units of cm/map units
    client: tdmclient
    node: node for the thymio robot
    '''
    def __init__(self, scaling_factor, client, node):
        # Saving tdmclient variables:
        self.node = node
        self.client = client

        # Saving dimensions of robots and environment
        self.scaling_factor = scaling_factor
        self.wheel_distance = 9.5 / self.scaling_factor  # 9.5 is the real distance in between the wheels in centimeters

        # We lock the node here instead of at every operation for speed issues. Locking and unlocking at every operation takes too much time.
        aw(self.node.lock())

    '''
    Reads the horizontal proximity sensors that are in the front of the thymio robot.
    It returns a list of 8 different integers, one for each proximity sensor.
    '''
    def read_sensors(self):
        aw(self.node.wait_for_variables({"prox.horizontal"}))
        aw(self.client.sleep(0.001)) # dummy sleep so that the read value is updated
        return list(self.node.v.prox.horizontal)

    '''
    Reads the linear speed of the thymio wheels in the units of the map per second.
    It returns a list of two floats, the linear speed of each wheel.
    The order is first left, then right.
    '''
    def read_speed(self):
        aw(self.node.wait_for_variables({"motor.left.speed", "motor.right.speed"}))
        aw(self.client.sleep(0.001)) # dummy sleep so that the read value is updated
        
        # The thymio returns an integer in [-500,500](thymio units/s), which corresponds 
        # to more or less [-20,20] cm/s (from the Thymio CheatSheet).
        return [
            self.node.v.motor.left.speed / 25.0 / self.scaling_factor,
            self.node.v.motor.right.speed / 25.0 / self.scaling_factor,
        ]

    '''
    Sets the target linear speed of the thymio wheels, in the thymio units [-500, 500].
    '''
    def set_speed(self, speed):
        speed_thymio = {
            "motor.left.target": [speed[0]],
            "motor.right.target": [speed[1]],
        }
        aw(self.node.set_variables(speed_thymio))

    def get_wheel_distance(self):
        return self.wheel_distance

    # Basic functionalities for thymio
    '''
    Locks thymio so that we can input and receive information.
    '''
    def lock(self):
        aw(self.node.lock())

    '''
    Unlocks the thymio so that it can be used in other clients (such as aseba studio).
    '''
    def unlock(self):
        aw(self.node.unlock())

    '''
    Stops and unlocks thymio.
    '''
    def terminate(self):
        self.set_speed((0, 0))
        self.unlock()
