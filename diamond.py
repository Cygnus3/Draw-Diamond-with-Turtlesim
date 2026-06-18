#!/usr/bin/env python3

import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from turtlesim.msg import Pose


ANGULAR_SPEED = 1.00   # rad/s while turning
TIMER_HZ      = 50    # 50 Hz control loop
TURN_ANGLE    = math.pi / 2          # 90 degrees in radians
ANGLE_TOL     = 0.005               # stop turning when within ~0.005 rad
DIST_TOL      = 0.01                 # stop moving when within 1 cm


def angle_diff(a, b):
    """Shortest signed difference from angle b to angle a, in (-pi, pi]."""
    d = a - b
    while d >  math.pi:
        d -= 2 * math.pi
    while d < -math.pi:
        d += 2 * math.pi
    return d


class DiamondMover(Node):

    def __init__(self):
        super().__init__('diamond_mover')

        self.declare_parameter('side',      3.0)   # side length of the diamond
        self.declare_parameter('speed',     1.0)
        self.declare_parameter('repeats',   1)
        self.declare_parameter('initangle', 45.0)  # degrees; 45° gives classic diamond

        self.side      = self.get_parameter('side').value
        self.speed     = self.get_parameter('speed').value
        self.repeats   = self.get_parameter('repeats').value
        initangle_deg  = self.get_parameter('initangle').value
        self.initangle = math.radians(initangle_deg)   # convert to radians

        # All 4 sides are equal — that's what makes it a diamond
        self.side_lengths = [self.side] * 4

        # current pose (updated by subscriber)
        self.pose: Pose | None = None

        # state machine
        self.phase       = 'wait_pose'
        self.side_index  = 0
        self.loops_done  = 0
        self.done        = False

        # reference values set at the start of each phase
        self.start_x      = 0.0
        self.start_y      = 0.0
        self.start_theta  = 0.0
        self.target_dist  = 0.0
        self.target_theta = 0.0

        self.pub   = self.create_publisher(Twist, '/turtle1/cmd_vel', 10)
        self.sub   = self.create_subscription(Pose, '/turtle1/pose', self._pose_cb, 10)
        self.timer = self.create_timer(1.0 / TIMER_HZ, self._tick)

        self.get_logger().info(
            f'Diamond mover ready — '
            f'side={self.side}, initangle={initangle_deg}°, speed={self.speed}'
        )

    # ------------------------------------------------------------------

    def _pose_cb(self, msg: Pose):
        self.pose = msg
        if self.phase == 'wait_pose':
            self._begin_align()   # first: rotate to initangle, then start moving

    def _stop(self):
        self.pub.publish(Twist())

    def _begin_align(self):
        """Rotate to initangle before drawing the first side."""
        self.phase        = 'align'
        self.target_theta = self.initangle
        self.get_logger().info(
            f'Aligning to initial heading {math.degrees(self.initangle):.1f}°'
        )

    def _begin_move(self):
        self.phase       = 'move'
        self.start_x     = self.pose.x
        self.start_y     = self.pose.y
        self.target_dist = self.side_lengths[self.side_index]
        self.get_logger().info(
            f'Moving side {self.side_index} — '
            f'distance {self.target_dist:.2f} units'
        )

    def _begin_turn(self):
        self.phase        = 'turn'
        self.start_theta  = self.pose.theta
        self.target_theta = self.start_theta + TURN_ANGLE
        while self.target_theta >  math.pi:
            self.target_theta -= 2 * math.pi
        while self.target_theta < -math.pi:
            self.target_theta += 2 * math.pi

    # ------------------------------------------------------------------

    def _turn_towards(self, target_theta):
        """Publish a turn command toward target_theta. Returns True when done."""
        err = angle_diff(target_theta, self.pose.theta)
        if abs(err) > ANGLE_TOL:
            factor = min(1.0, abs(err) / (0.2 * TURN_ANGLE + 1e-6))
            msg = Twist()
            msg.angular.z = math.copysign(max(0.15, ANGULAR_SPEED * factor), err)
            self.pub.publish(msg)
            return False
        self._stop()
        return True

    def _tick(self):
        if self.pose is None or self.done:
            return

        msg = Twist()

        if self.phase == 'align':
            if self._turn_towards(self.target_theta):
                self._begin_move()

        elif self.phase == 'move':
            travelled = math.hypot(
                self.pose.x - self.start_x,
                self.pose.y - self.start_y)
            remaining = self.target_dist - travelled

            if remaining > DIST_TOL:
                msg.linear.x = self.speed
                self.pub.publish(msg)
            else:
                self._stop()
                self.side_index = (self.side_index + 1) % 4
                self._begin_turn()

        elif self.phase == 'turn':
            if self._turn_towards(self.target_theta):
                if self.side_index == 0:
                    self.loops_done += 1
                    self.get_logger().info(f'Diamond {self.loops_done} complete.')
                    if self.repeats != -1 and self.loops_done >= self.repeats:
                        self.get_logger().info('All diamonds complete. Stopping.')
                        self.done = True
                        self.timer.cancel()
                        return
                self._begin_move()


# ----------------------------------------------------------------------

def main(args=None):
    rclpy.init(args=args)
    node = DiamondMover()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()