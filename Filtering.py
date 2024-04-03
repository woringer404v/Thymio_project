import functools
import jax
import jax.numpy as jnp

# Source: https://github.com/google/jax/pull/762#issuecomment-1002267121
# Implements a function that computes both the vector value and jacobian
# of a function using the jax auto-diff tools
# Jax auto-diff tools used to make it the code more easily used
def value_and_jacfwd(f, x):
    pushfwd = functools.partial(jax.jvp, f, (x,))
    basis = jnp.eye(x.size, dtype=x.dtype)
    y, jac = jax.vmap(pushfwd, out_axes=(None, 1))((basis,))
    return y, jac


class PredictModule(object):
    '''
    Basic state predictor that just outputs the previous state as the prediction 
    (no update).
    '''
    def __init__(self):
        return

    # Identity function as prediction. No control or time is used.
    def predict(self, x, u, t):
        return x


class DifferentialDrive(PredictModule):

    '''
    State predictor for the case of a differential drive wheeled robot, as, 
    for example, the Thymio Robot.

    In here, the state is (x,y,theta), and we assume that we get as control input
    the speed of the wheels.

    We use the jax library to easily compute derivatives.
    '''

    '''
    wheel_separation: The distance in between the two turning wheels of the differential robot.
    predict_uncertainty: Two dimensional iterable of size 
    '''
    def __init__(
        self,
        wheel_separation,
        predict_uncertainty,
        control_uncertainty,
    ):
        '''
        Args:
            wheel_separation (float): The distance in between the two turning wheels of the differential robot in map units.
            predict_uncertainty (np.array): covariance matrix of assumed gaussian noise of our prediction
            control_uncertainty (np.array): covariance matrix of assumed gaussian noise in the readings of the wheel speeds
        '''
        self.wheel_separation_ = wheel_separation
        # this should probably not be here as it is not something intrinsic to a PredictModule
        self.Q_ = jnp.array(predict_uncertainty)
        self.U_ = jnp.array(
            control_uncertainty
        )  

    def get_Q(self):
        '''
        Get predict covariance error (given control)
        '''
        return self.Q_

    def get_U(self):
        '''
        Get control variables covariance error
        '''
        return self.U_

    
    def predict(self, x, u, t):
        '''
        Predict function that is based on the kinematics of the differential 
        wheeled robot, where we are assuming that the control variables (speed
        of the wheels) have remained constant for the whole duration t.
        
        In u we have [vr, vl] if x and y follow the right hand rule to compute 
        the angle, if they don't, it should be [vl, vr].
        '''
        V = (u[0] + u[1]) / 2
        w = (u[0] - u[1]) / self.wheel_separation_
        angle = x[2] * (jnp.pi / 180)
        if abs(w) <= 1e-5:
            increment = jnp.array([V * t * jnp.cos(angle), V * t * jnp.sin(angle), 0])
        else:
            increment = jnp.array(
                [
                    V / w * (jnp.sin(t * w + angle) - jnp.sin(angle)),
                    -V / w * (jnp.cos(t * w + angle) - jnp.cos(angle)),
                    w * t * 180 / jnp.pi,
                ]
            )
        return x + increment

    def predict_and_jacobian(self, x, u, t):
        '''
        Returns the prediction f(x) and the jacobians in f(x) with respect 
        to:
        1. The previous state variables x
        2. The control variables u.
        This implicitly defines the linearized function around f(x), which will 
        be used to propagate uncertainties.
        '''
        xp, J = value_and_jacfwd(
            lambda q: self.predict(q[0:3], q[3:5], q[5]), jnp.concatenate([x, u, t])
        )
        return xp, J[:, 0:3], J[:, 3:5]


class MeasurementModule(object):
    '''
    Module that defines the transformation function h(x) that goes from the state 
    space X to the measurement space Z.

    Assuming small uncertainties, the expected measurement of a state x will be h(x).

    This one is the most basic one, where the state space and the measurement
    one is the same, so the function is just the identity.

    We use jax to easily get derivatives. Even though in this case it is just 
    the identity, we leave the jax function so that it is easily modifiable in 
    the case the measurement space changes.
    '''
    def __init__(self, measurement_error):
        '''
        Args:
            measurement_error (np.array): Covariance matrix of the assumed gaussian noise of the measurement
        '''
        self.R_ = jnp.array(measurement_error)
        return

    def get_R(self):
        '''
        Get measurement covariance error
        '''
        return self.R_

    def expected_measure(self, x):
        '''
        h function, identity in this case
        '''
        return x

    def expected_measure_and_jacobian(self, x):
        '''
        Returns h(x) and the jacobian with respect to the state x
        '''
        return value_and_jacfwd(lambda q: self.expected_measure(q), x)


class ExtendedKalmanFilter(object):
    '''
    Extended Kalman Filter Class
    The prediction and measurement function have been separated into different 
    modules, so that it is easily adapted to new tasks.
    '''
    def __init__(self, predict_module, measurement_module):
        '''
        Args:
            predict_module (Class): class like DifferentialDrive
            measurement_module (Class): class like MeasurementModule
        '''
        self.predict_module_ = predict_module
        self.measurement_module_ = measurement_module
        self.previous_position_ = None
        self.previous_uncertainty_ = None
        self.previous_time = None


    def predict(self, x, u, t, P):
        '''
        Predict Step of the Kalman Filter
        '''
        xp, F, B = self.predict_module_.predict_and_jacobian(x, u, t)
        Pp = (
            F @ P @ F.T
            + B @ self.predict_module_.get_U() @ B.T
            + self.predict_module_.get_Q()
        )

        return xp, Pp

    def update(self, xp, Pp, z, R=None):
        '''
        Update Step of the Extended Kalman Filter
        '''
        h, H = self.measurement_module_.expected_measure_and_jacobian(xp)

        y = z - h
        # if we are passed the measurement error, we use it instead of 
        # the saved default one
        if R:
            S = H @ Pp @ H.T + R
        else:
            S = H @ Pp @ H.T + self.measurement_module_.get_R()
        K = Pp @ H.T @ jnp.linalg.inv(S)
        x = xp + K @ y
        P = (jnp.eye(len(x)) - K @ H) @ Pp
        return x, P

    def iterate(self, u, t, z, R=None):
        '''
        Full Step of the Kalman Filter: Predict + Measurement.
        If there is no given measurement, it will just apply the predict update.
        We save the state estimate and uncertainty in the object, so it is not 
        needed as a parameter in the function.
        '''
        x = self.previous_position_
        P = self.previous_uncertainty_

        xp, Pp = self.predict(
            jnp.array(x),
            jnp.array(u),
            jnp.array([t - self.previous_time]),
            jnp.array(P),
        )

        if z is not None:
            if R:
                xu, Pu = self.update(xp, Pp, jnp.array(z), jnp.array(R))
            else:
                xu, Pu = self.update(xp, Pp, jnp.array(z))
        else:
            xu, Pu = xp, Pp
    
        self.previous_time = t
        self.previous_position_ = xu.copy()
        self.previous_uncertainty_ = Pu.copy()

        return xu

    def initialize(self, initial_time, initial_position, initial_uncertainty=None):
        '''
        Give an initial value to the position, uncertainty and time.
        '''
        self.previous_time = initial_time
        self.previous_position_ = jnp.array(initial_position)
        if initial_uncertainty is None:
            self.previous_uncertainty_ = self.measurement_module_.get_R().copy()
        else:
            self.previous_uncertainty_ = jnp.array(initial_uncertainty.copy())

def create_filter(wheel_distance, predict_std=0, control_std=1, measure_std=0.25):
    '''
    Create an EKF filter for a differential drive wheeled robot that has a direct 
    measurement on x, y and theta.
    '''
    cov_predict = predict_std**2 * jnp.eye(3)
    cov_control = control_std**2 * jnp.eye(2)
    cov_measure = measure_std**2 * jnp.eye(3)
    predict = DifferentialDrive(wheel_distance, cov_predict, cov_control)
    measurement = MeasurementModule(cov_measure)
    filter = ExtendedKalmanFilter(predict, measurement)

    return filter
